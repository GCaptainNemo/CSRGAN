# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# #
# # criterion = nn.BCELoss()
#
#
# class SoftConditionCriterion(nn.Module):
#     """
#     soften conditional GAN discriminator loss function
#     """
#     def __init__(self):
#         super(SoftConditionCriterion, self).__init__()
#         self.bce_loss = nn.BCELoss()
#
#     def forward(self, predicted_phy, true_phy, predicted_prop_sr):
#         # N x 1 distance
#         euclidean_distance = F.pairwise_distance(predicted_phy, true_phy, keepdim=True)
#
#         nn.Sigmoid()()
#         # postive sample label = 0 distance descend
#         # negative sample label = 1
#         # negative sample distance has lower bound self.margin
#         loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
#                       label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         # cosine_similarity = torch.cosine_similarity(output1, output2, dim=1)
#         # loss = torch.mean((cosine_similarity - label) ** 2)
#         return loss
# #
# #
# #
