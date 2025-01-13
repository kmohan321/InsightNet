import torch
import torch.nn as nn

class Rope(nn.Module):
  def __init__(self, max_seq_length,head_dim):
    super().__init__()
    
    assert head_dim%2==0, 'head_dim should be even'
    
    # m values for different positions in sequence
    m = torch.arange(0,max_seq_length)
    
    # theta values for different index in token vector
    theta = 1/(10000**(2*torch.arange(0,head_dim//2)/head_dim))
    
    #all possible combinations for m and theta
    freq = torch.outer(m,theta)
    
    #converting freq to polar
    complex_freq = torch.polar(torch.ones_like(freq),freq)
    
    self.register_buffer('complex_freq', complex_freq.unsqueeze(0).unsqueeze(2))
    
  def forward(self):
        return self.complex_freq

def apply_rope(x,complex_freq):
    b ,s,  d = x.shape
    x = x.view(b, s, -1, 2)
    x = torch.view_as_complex(x)
 
    x = x * complex_freq[:,:s,:].squeeze(2)
    x = torch.view_as_real(x)
    x = x.view(b,s,d)
    return x
    

class Patchifier(nn.Module):
  def __init__(self,
               in_channels:int,
               height:int,
               hidden_dim:int,
               patch_size:int
               ):
    super().__init__()
    self.hidden_dim = hidden_dim
    
    self.conv = nn.Conv2d(in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
    self.num_patches = (height//patch_size) **2
    
  def forward(self,x,complex_freq):
    x = self.conv(x)
    x = x.permute(0,2,3,1).contiguous().view(-1,self.num_patches ,self.hidden_dim)
    x = apply_rope(x,complex_freq)
    return x

class MHA(nn.Module):
  def __init__(self,
               hidden_dim:int,
               num_heads:int,
               head_dim:int):
    super().__init__()
    
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.head_dim = head_dim
    
    self.Wq = nn.Linear(hidden_dim,hidden_dim,bias=False)
    self.Wk = nn.Linear(hidden_dim,hidden_dim,bias=False)
    self.Wv = nn.Linear(hidden_dim,hidden_dim,bias=False)
    self.Wo = nn.Linear(hidden_dim,hidden_dim,bias=False)
    
    self.scale = self.head_dim ** -0.5
    
  def forward(self,x):
    
    b,s,d = x.shape
    #(b,s,d) -> (b,s,n_h,h_d)
    q,k = self.Wq(x).view(-1,s,self.num_heads,self.head_dim), self.Wk(x).view(-1,s,self.num_heads,self.head_dim)
    v = self.Wv(x).view(-1,s,self.num_heads,self.head_dim)
    
    #(b,s,n_h,h_d) -> (b,h_d,s,h_d)
    q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
    
    attention_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
    attention_weights = torch.softmax(attention_scores, dim=-1)
    out = (attention_weights @ v).transpose(1,2).contiguous().view(-1,s,self.hidden_dim)
    return self.Wo(out)
    
    
class Encoder_blocks(nn.Module):
  def __init__(self,
               hidden_dim: int,
               num_heads: int,
               head_dim: int,
               mlp_multiplier: int
               ):
    super().__init__()
    
    self.layer_norm1 = nn.LayerNorm(hidden_dim)
    self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    self.mha = MHA(hidden_dim,num_heads,head_dim)
    self.ffn = nn.Sequential(
      nn.Linear(hidden_dim,hidden_dim * mlp_multiplier),
      nn.GELU(),
      nn.Linear(hidden_dim * mlp_multiplier , hidden_dim)
    )
  def forward(self,x):
    x = x + self.mha(self.layer_norm1(x))
    x = x + self.ffn(self.layer_norm2(x))
    return x
    

class VisualEncoder(nn.Module):
  def __init__(self,
                in_channels:int,
                height:int,
                patch_size:int,
                num_blocks: int,
                hidden_dim:int,
                max_seq_length:int,
                num_heads:int,
                head_dim : int,
                mlp_multiplier: int
                ):
    super().__init__()
    
    self.rope = Rope(max_seq_length,hidden_dim)
    self.patchifier = Patchifier(in_channels, height,hidden_dim, patch_size)
    self.encoder_blocks = nn.ModuleList([Encoder_blocks(hidden_dim,num_heads,head_dim,mlp_multiplier) 
                                         for _ in range(num_blocks)])
    
  def forward(self, x):
    
    complex_freq = self.rope()
    x = self.patchifier(x,complex_freq)
    
    for block in self.encoder_blocks:
      x = block(x)
    return x
    
