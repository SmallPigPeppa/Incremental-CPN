from utils import get_pretrained_encoder, get_cifar10_test_loader
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    ckpt_path = 'pretrained_model/simclr-cifar100-265s7gks-ep=999.ckpt'
    encoder = get_pretrained_encoder(ckpt_path=ckpt_path)
    cifar10_test_loader = get_cifar10_test_loader()
    for batch in cifar10_test_loader:
        a=batch
    # s_cos = cosine_similarity(X=sx, dense_output=True)
    # s2_cos = cosine_similarity(X=sx2, dense_output=True)
    # ssl_cos = cosine_similarity(X=sslx, dense_output=True)
#
#
#     s_cos = cosine_similarity(X=sx, dense_output=True)
#     s2_cos = cosine_similarity(X=sx2, dense_output=True)
#     ssl_cos = cosine_similarity(X=sslx, dense_output=True)
#
# # sort samplers by label
# sx = np.load('./s_test_x.npy')
# sx2 = np.load('./s_test0.npy')
# sy = np.load('./s_test_y.npy')
# sslx = np.load('./ssl_test_test.npy')
# ssly = np.load('./y_test0.npy')
# sslorder = ssly.argsort()
# sorder = sy.argsort()
# sslx = sslx[sslorder]
# sx = sx[sorder]
# sx2 = sx2[sorder]
# print(ssly)
# print(sorder)
# print(ssly[sslorder])
# num_samplers = 10000
# a1 = np.ones(shape=(10000, 10000))
# a0 = np.zeros(shape=(1000, 1000))
# for i in range(10):
#     a1[i * 1000:(i + 1) * 1000, i * 1000:(i + 1) * 1000] = a0
#
# from sklearn.metrics.pairwise import cosine_similarity
#
# print(ssly.shape)
# print(sslx.shape)
# print(sx.shape)
# s_cos = cosine_similarity(X=sx, dense_output=True)
# s2_cos = cosine_similarity(X=sx2, dense_output=True)
# ssl_cos = cosine_similarity(X=sslx, dense_output=True)
# # ssl_cos=ssl_cos-a1*0.02
# # s_cos=s_cos+0.003*a1
# print(s_cos.shape)
# print(ssl_cos)
# np.save('./s_cos', s_cos)
# np.save('./ssl_cos', ssl_cos)
#
# fig, ax = plt.subplots()
# plt.imshow(ssl_cos, cmap=plt.cm.Blues)
# cbar = plt.colorbar()
# ax.tick_params(axis='both', which='major', labelsize=13)
# plt.clim(0, 1)
# for t in cbar.ax.get_yticklabels():
#     t.set_fontsize(13)
# fig.savefig('ssl_cos.png', format='png', dpi=300)
# plt.show()
#
# # fig, ax = plt.subplots()
# # plt.imshow(s_cos,cmap=plt.cm.Blues)
# # cbar=plt.colorbar()
# # ax.tick_params(axis='both', which='major', labelsize=13)
# # plt.clim(0, 1)
# # for t in cbar.ax.get_yticklabels():
# #      t.set_fontsize(13)
# # # plt.savefig('s_cos.pdf')
# # fig.savefig('s_cos.png', format='png', dpi=500)
# # plt.show()
