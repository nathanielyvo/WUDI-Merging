import torch
from tqdm import tqdm
def get_merge_vector(vectors, iter_num = 300):

    vectors = vectors.cuda() # vectors.shape = [8, 2304, 768]

    merging_vector = torch.nn.Parameter((torch.sum(vectors, dim = 0)))

    optimizer = torch.optim.Adam([merging_vector], lr=2e-5)

    l2_norms = torch.square(torch.norm(vectors.reshape(8, -1), p=2, dim=1))
    print(l2_norms)

    for i in tqdm(range(iter_num)):

        inner_product = torch.matmul( merging_vector.unsqueeze(0) - vectors , vectors.transpose(1, 2)) 
        loss =  torch.sum( torch.square(inner_product ) / l2_norms.unsqueeze(-1).unsqueeze(-1) )

        optimizer.zero_grad()          
        loss.backward()
        optimizer.step()

    return merging_vector.data.detach().cpu()
