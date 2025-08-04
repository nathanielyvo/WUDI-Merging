
import torch


def calculate_cosine_loss(specific_vectors):
    loss_original = 0
    for i in range(specific_vectors.shape[1]):
        vectors_i = specific_vectors[:, i, :]  # [100, 128]
        similarity_matrix = torch.nn.functional.cosine_similarity(
            vectors_i.unsqueeze(1), vectors_i.unsqueeze(0), dim=2
        )
        similarity_matrix = torch.tril(similarity_matrix, diagonal=-1)
        loss_original += similarity_matrix.abs().mean()
    loss_original = loss_original / specific_vectors.shape[1]
    return loss_original


def calculate_cosine_loss_function(specific_vectors):
    # specific_vectors.shape = [8, 2304, 768]
    eps = 1e-8
    sv = specific_vectors.permute(1, 0, 2)
    norms = torch.norm(sv, dim=2, keepdim=True) + eps
    sv_norm = sv / norms # [2304, 8, 768]
    cosine_sim = torch.matmul(sv_norm, sv_norm.transpose(1, 2))  # [2304, 8, 8]
    similarities_mask = torch.tril(torch.ones_like(cosine_sim[0]), diagonal=-1)
    similarities = cosine_sim * similarities_mask
    loss = torch.sum(torch.abs(similarities)) / (torch.sum(similarities_mask) * cosine_sim.shape[0])
    return loss


def get_common_task_vector(vectors, split, l1_coef):
    # l1_coef=0.001
    vectors.requires_grad = True
    print("get common vector l1_coef = ", l1_coef)
    vectors = vectors.cuda() # vectors.shape = [8, 2304, 768]
    # print("get common vector vectors.shape = ", vectors.shape)
    shape = vectors.shape
    common_vector_param = torch.nn.Parameter(torch.zeros_like(vectors[0]).cuda())
    optimizer = torch.optim.SGD([common_vector_param], lr=1e-5, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.Adam([common_vector_param], lr=1e-5)
    cosine_before = 0
    cosine_after = 0
    for i in range(2000):
        specific_vectors = vectors - common_vector_param # specific_vectors.shape = [8, 2304, 768]
        if split == 'row':
            cosine_loss = calculate_cosine_loss_function(specific_vectors)
        elif split == 'column':
            cosine_loss = calculate_cosine_loss_function(specific_vectors.permute(1, 0, 2))
        elif split == 'none':
            specific_vectors = specific_vectors.view(shape[0], -1)
            similarity_matrix = F.cosine_similarity(specific_vectors.unsqueeze(1), specific_vectors.unsqueeze(0), dim=2)
            similarity_matrix = torch.tril(similarity_matrix, diagonal=-1)
            cosine_loss = similarity_matrix.abs().mean()
        else:
            raise ValueError('Invalid split')
        # value_loss = torch.sum(torch.abs(common_vector_param))
        value_loss = torch.sum(torch.norm(common_vector_param, dim=1))
        loss = cosine_loss + value_loss * l1_coef
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()
        if i == 0:
            print("before cosine_loss = ", cosine_loss.item())
            print("before value_loss = ", value_loss.item())
            cosine_before = cosine_loss.item()
    print("after cosine_loss = ", cosine_loss.item())
    print("after value_loss = ", value_loss.item())
    cosine_after = cosine_loss.item()
    common_vector = common_vector_param.data.detach().cpu()
    return common_vector, cosine_before, cosine_after
