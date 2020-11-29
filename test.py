import torch
import model
import pickle
import numpy as np

vocab_info = pickle.load(open('data/nlp_character_vocab.pkl', 'rb+'))
test = "Thành phố Hà Nội Quận Đố"
test = test.lower()
encode = [vocab_info[c][0] for c in test]
print(encode)

_model = model.SimpleLSTM(embedding_dim=512, hidden_dim=50, vocab_size=len(vocab_info), tagset_size=len(vocab_info))
_model = torch.nn.DataParallel(_model).cuda()
checkpoint = torch.load('31.pth')['state_dict']
_model.load_state_dict(checkpoint)
_model.eval()
output = torch.nn.functional.softmax(_model(torch.tensor([encode])), dim=1)
output = output.cpu().detach().numpy()
print(np.amax(output))
for k, v in vocab_info.items():
    if v[0] == np.argmax(output):
        print(test+k)
        break