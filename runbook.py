import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
from electricity import get_data
from bigramnn import Model, batch_size, block_size, device
import numpy as np

#from baselines import MeanPredictor
#from tsmamba import Model
#hyperparams
lr = 1e-3
max_iters = 50 # Default: 10000
print_iters = 50
eval_iters = 10
eval_interval = 300
# ---------

raw, encode, decode, vocab_size = get_data(2)
modal_token = encode(['0.12'])[0]
print(f'Modal token is {modal_token}')
baseline_logits = torch.zeros((batch_size*block_size, vocab_size))
baseline_logits[:,modal_token] = 1
baseline_logits = baseline_logits.to(device)
print(baseline_logits)
# train and test splits
data = torch.tensor(encode(raw), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  # generate targets and context
  if split == "train":
    data = train_data
  else:
    data = val_data
  index = torch.randint(0,len(data)-block_size,(batch_size,))
  x = torch.stack([data[ind:ind+block_size] for ind in index])
  y = torch.stack([data[ind+1:ind+block_size+1] for ind in index])
  return x.to(device),y.to(device)

first_batch = True
@torch.no_grad()
def estimate_loss():
  global first_batch
  out = {}
  model.eval()
  for split in ['train', 'test']:
    losses = torch.zeros(eval_iters)
    baseline_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)
      logits, loss = model(X,Y)
      if first_batch:
        print('batch X,Y', X.shape, Y.shape)
        print(X)
        print(Y)
        print('logits', logits.shape, logits)
        print('loss', loss.shape, loss)
        first_batch = False
      losses[k] = loss.item()
      baseline_loss = nn.functional.cross_entropy(baseline_logits, Y.view(batch_size*256))
      baseline_losses[k] = baseline_loss.item()
    out[split] = losses.mean()
    out['baseline'] = baseline_losses.mean()
  model.train()
  return out

model = Model(vocab_size)
#baseline = MeanPredictor()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

# checkpoint = torch.load('model.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
checkpoint_path = None#"./differentattention/model_40.pt"
epoch = 0
if checkpoint_path:
  checkpoint = torch.load(checkpoint_path)
  print(checkpoint)
  if checkpoint['model_state_dict']:
    model.load_state_dict(checkpoint['model_state_dict'].to(device))
  if checkpoint['optimizer_state_dict']:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
device = "cuda"
m = model.to(device)
print("Uses device " + device)
MODEL_CHECKPOINT = "./differentattention/model_{iter}.pt"
losses_data = {"train":[], "test":[], "baseline":[]}
for iter in tqdm(range(epoch ,max_iters)):
  if iter % eval_iters == 0:
    losses = estimate_loss()
    losses_data['train'].append(losses['train'].cpu().numpy())
    losses_data['test'].append(losses['test'].cpu().numpy())
    losses_data['baseline'].append(losses['baseline'].cpu().numpy())
    print(f"Step {iter}, train loss:{losses['train']:.4f}, test loss:{losses['test']:.4f}, baseline loss:{losses['baseline']:.4f}")

  if iter % print_iters == 0:
    losses = estimate_loss()
    torch.save({
            'epoch': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            }, MODEL_CHECKPOINT.format(iter=iter))
    losses_data['train'].append(losses['train'].cpu().numpy())
    losses_data['test'].append(losses['test'].cpu().numpy())
    losses_data['baseline'].append(losses['baseline'].cpu().numpy())
    model.eval()
    with torch.no_grad():
      #Generate from the model:
      output = m.generate(torch.zeros((1,2), dtype=torch.long).to(device).contiguous(), 1000)
      for arr in output:
        print(decode(arr.cpu().detach().numpy()))
    print(f"Step {iter}, train loss:{losses['train']:.4f}, test loss:{losses['test']:.4f}, baseline loss:{losses['baseline']:.4f}")
    model.train()

  #Get data
  xb,yb = get_batch("train")

  #Evaluate loss
  logits,loss = model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
  optimizer.step()
torch.save(model.state_dict(), "./differentattention/model.pt")

#Generate from the model:

pred_size = 24*10

print('RAW GENERATION')
output = m.generate(torch.zeros((1,2), dtype=torch.long).to(device).contiguous(), pred_size)

for arr in output:
    print(decode(arr.cpu().detach().numpy()))
    
print('PREDICTION W/O HINTS')
prefix = val_data[:block_size]

output = m.generate(torch.stack([prefix]).to(device), pred_size)
print('Preds')
for arr in output:
    print(decode(arr.cpu().detach().numpy()))
print('Ground truth')
print(decode(val_data[:block_size+pred_size].cpu().detach().numpy()))

pred_length = 24
print('PREDICTION W HINTS')
prefix = val_data[:block_size+pred_size]
output = val_data[:block_size].cpu().detach().numpy()
for i in range(0, pred_size, pred_length):
  preds = m.generate(torch.stack([prefix[i:i+block_size]]).to(device), pred_length)
  output = np.append(output, preds.cpu().detach().numpy()[0][-pred_length:])
output = decode(output)
print('Preds')
print(output)
print('Ground truth')
print(decode(val_data[:block_size+pred_size].cpu().detach().numpy()))