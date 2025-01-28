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
    b ,s, h,d = x.shape
    x = x.view(b, s, h,-1, 2)
    x = torch.view_as_complex(x)
 
    x = x * complex_freq[:,:s,:,:]
    x = torch.view_as_real(x)
    x = x.view(b,s,h,d)
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
    
  def forward(self,x,rope_freq):
    
    b,s,d = x.shape
    #(b,s,d) -> (b,s,n_h,h_d)
    q,k = self.Wq(x).view(-1,s,self.num_heads,self.head_dim), self.Wk(x).view(-1,s,self.num_heads,self.head_dim)
    v = self.Wv(x).view(-1,s,self.num_heads,self.head_dim)
    
    #applying the rope values
    q,k = apply_rope(q,rope_freq),apply_rope(k,rope_freq)
    #(b,s,n_h,h_d) -> (b,h_d,s,h_d)
    q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
    
    attention_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
    attention_weights = torch.softmax(attention_scores, dim=-1)
    out = torch.einsum('bhss,bhsd->bhsd',attention_weights,v).transpose(1,2).contiguous().view(-1,s,self.hidden_dim)
    
    return self.Wo(out)

class Encoder(nn.Module):
  def __init__(self,
               hidden_dim: int,
               num_heads:int,
               head_dim:int,
               mlp_mlt:int
               ):
    super().__init__()
    
    self.mha = MHA(hidden_dim,num_heads,head_dim)
    self.layer_norm1 = nn.LayerNorm(hidden_dim)
    self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    self.ffn =nn.Sequential(
        nn.Linear(hidden_dim,mlp_mlt*hidden_dim),
        nn.GELU(),
        nn.Linear(mlp_mlt*hidden_dim,hidden_dim)
       )
    
  def forward(self,x,rope_freq):
    x = x + self.layer_norm1(self.mha(x,rope_freq))
    x = x + self.layer_norm2(self.ffn(x))
    return x

class TextEncoder(nn.Module):
  def __init__(self,
                num_blocks:int,
                hidden_dim: int,
                num_heads:int,
                head_dim:int,
                mlp_mlt:int,
                max_seq_length:int
               ):
    super().__init__()
    
    self.blocks = nn.ModuleList([Encoder(hidden_dim,num_heads,head_dim,mlp_mlt) for _ in range(num_blocks)])
    self.rope = Rope(max_seq_length,head_dim)
    
  def forward(self,x):

      complex_freq = self.rope()
      for block in self.blocks:
        x = block(x,complex_freq)
      return x
 
  

    
    