import torch

from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn import *
from torch.optim import Adam
from torch.nn.init import uniform_

class BPR(Module):
    def __init__(self, user_set:iter, item_set:iter, hidden_dim=512):
        super(BPR, self).__init__()
        self.hidden_dim = hidden_dim
       
        self.user_gama = Embedding(len(user_set), self.hidden_dim)
        self.item_gama = Embedding(len(item_set), self.hidden_dim)
        self.user_beta = Embedding(len(user_set), 1)
        self.item_beta = Embedding(len(item_set), 1)

        self.user_set = list(user_set)
        self.item_set = list(item_set)
        
        init.uniform_(self.user_gama.weight, 0, 0.01)
        init.uniform_(self.user_beta.weight, 0, 0.01)
        init.uniform_(self.item_gama.weight, 0, 0.01)
        init.uniform_(self.item_beta.weight, 0, 0.01)
        
        self.user_idx = {user:_index for _index, user in enumerate(user_set)}
        self.item_idx = {item:_index for _index, item in enumerate(item_set)}

    def get_user_idx(self, users):
        if self.user_beta.weight.is_cuda:
            return torch.tensor([self.user_idx[user] for user in users]) \
                .long() \
                .cuda(self.user_beta.weight.get_device())
        else:
            return torch.tensor([self.user_idx[user] for user in users]) \
                .long()
    def get_item_idx(self, items):
        if self.user_beta.weight.is_cuda:
            return torch.tensor([self.item_idx[item] for item in items]) \
                .long() \
                .cuda(self.user_beta.weight.get_device())
        else:
            return torch.tensor([self.item_idx[item] for item in items]) \
                .long()
    def get_user_gama(self, users):
        return self.user_gama(self.get_user_idx(users))
    def get_item_gama(self, items):
        return self.item_gama(self.get_item_idx(items))
    def forward(self, users, items):
        batchsize = len(users)
        user_gama = self.get_user_gama(users)
        user_beta = self.user_beta(self.get_user_idx(users))
        item_gama = self.get_item_gama(items)
        item_beta = self.item_beta(self.get_item_idx(items))
        return item_beta.view(batchsize) + user_beta.view(batchsize) \
            + bmm(user_gama.view(batchsize, 1, self.hidden_dim), 
                item_gama.view(batchsize, self.hidden_dim, 1)).view(batchsize)

    def fit(self, users, items, p=2):
        batchsize = len(users)
        user_gama = self.get_user_gama(users)
        user_beta = self.user_beta(self.get_user_idx(users))
        item_gama = self.get_item_gama(items)
        item_beta = self.item_beta(self.get_item_idx(items))
        return item_beta.view(batchsize) + user_beta.view(batchsize) \
            + bmm(user_gama.view(batchsize, 1, self.hidden_dim),
                item_gama.view(batchsize, self.hidden_dim, 1)).view(batchsize), \
            user_gama.norm(p=p)+ item_beta.norm(p=p)+ user_beta.norm(p=p)+item_gama.norm(p=p)

class VTBPR(BPR):
    def __init__(self, user_set, item_set, hidden_dim=512,
        theta_text = True, theta_visual = True):
        super(VTBPR, self).__init__(user_set, item_set, hidden_dim=hidden_dim)

        self.theta_user_visual = Embedding(len(user_set), self.hidden_dim)
        self.theta_user_text = Embedding(len(user_set), self.hidden_dim)

        init.uniform_(self.theta_user_text.weight, 0, 0.01)
        init.uniform_(self.theta_user_visual.weight, 0, 0.01)

    def get_theta_user_visual(self, users):
        return self.theta_user_visual(self.get_user_idx(users))

    def get_theta_user_text(self, users):
        return self.theta_user_text(self.get_user_idx(users))

    def forward(self, users, items, visual_features, textural_features):
        batchsize = len(users)
        bpr = BPR.forward(self, users, items)
        theta_user_visual = self.get_theta_user_visual(users)
        theta_user_text = self.get_theta_user_text(users)
        
        return bpr \
            + bmm(theta_user_visual.view(batchsize, 1, self.hidden_dim), 
                visual_features.view(batchsize, self.hidden_dim , 1)).view(batchsize) \
            + bmm(theta_user_text.view(batchsize, 1, self.hidden_dim), 
                textural_features.view(batchsize, self.hidden_dim, 1 )).view(batchsize)
         
    def fit(self, users, items, visual_features, textural_features):
        batchsize = len(users)
        bpr, bprweight = BPR.fit(self, users, items)

        theta_user_visual = self.get_theta_user_visual(users)
        theta_user_text = self.get_theta_user_text(users)
        
        return bpr \
            + bmm(theta_user_visual.view(batchsize, 1, self.hidden_dim), 
                visual_features.view(batchsize, self.hidden_dim , 1)).view(batchsize) \
            + bmm(theta_user_text.view(batchsize, 1, self.hidden_dim), 
                textural_features.view(batchsize, self.hidden_dim, 1 )).view(batchsize), \
            bprweight + self.get_theta_user_text(set(users)).norm(p=2) + self.get_theta_user_visual(set(users)).norm(p=2)

class TextCNN(Module):
    def __init__(self, sentence_size = (83, 300), output_size = 512, uniform=False):
        super(TextCNN, self).__init__()
        self.max_sentense_length, self.word_vector_size = sentence_size
        self.text_cnn = ModuleList([Sequential(
            Conv2d(in_channels=1,out_channels=100,kernel_size=(2,self.word_vector_size),stride=1),
            Sigmoid(),
            MaxPool2d(kernel_size=(self.max_sentense_length - 1,1),stride=1)
        ), Sequential(
            Conv2d(in_channels=1,out_channels=100,kernel_size=(3,self.word_vector_size),stride=1),
            Sigmoid(),
            MaxPool2d(kernel_size=(self.max_sentense_length - 2,1),stride=1)
        ), Sequential(
            Conv2d(in_channels=1,out_channels=100,kernel_size=(4,self.word_vector_size),stride=1),
            Sigmoid(),
            MaxPool2d(kernel_size=(self.max_sentense_length - 3,1),stride=1)
        ), Sequential(
            Conv2d(in_channels=1,out_channels=100,kernel_size=(5,self.word_vector_size),stride=1),
            Sigmoid(),
            MaxPool2d(kernel_size=(self.max_sentense_length - 4,1),stride=1)
        )])
        self.text_nn = Sequential(
            Linear(400,output_size),
            Sigmoid(),
        )
        if uniform == True:
            for i in range(4):
                init.uniform_(self.text_cnn[i][0].weight.data, 0, 0.001)
                init.uniform_(self.text_cnn[i][0].bias.data, 0, 0.001)
            init.uniform_(self.text_nn[0].weight.data, 0, 0.001)
            init.uniform_(self.text_nn[0].bias.data, 0, 0.001)
    def forward(self, input):
        return self.text_nn(
            cat([conv2d(input).squeeze_(-1).squeeze_(-1) for conv2d in self.text_cnn], 1)
        )

class GPBPR(Module):
    def __init__(self, user_set, item_set, embedding_weight ,
        max_sentence = 83,  text_feature_dim=300, 
        visual_feature_dim = 4096, hidden_dim=512,
        uniform_value = 0.5):
        
        super(GPBPR, self) .__init__()
        self.epoch = 0
        self.uniform_value = uniform_value
        self.hidden_dim = hidden_dim
        # print(list(self.features.keys()))
        self.visual_nn = Sequential(
            Linear(visual_feature_dim, self.hidden_dim),
            Sigmoid(),
        )
        self.visual_nn[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.visual_nn[0].apply(lambda module: uniform_(module.bias.data,0,0.001))

        print('generating user & item Parmeters')
        
        # load text features
        self.max_sentense_length = max_sentence

        # text embedding layer
        self.text_embedding = Embedding.from_pretrained(embedding_weight, freeze=False)

        '''
            text features embedding layers
        '''
        self.vtbpr = VTBPR(user_set=user_set, item_set=item_set, hidden_dim=self.hidden_dim)
        self.textcnn = TextCNN(sentence_size=(max_sentence,text_feature_dim), output_size=hidden_dim)
        print('Module already prepared, {} users, {} items'.format(len(user_set), len(item_set)))

    def forward(self, batch, visual_features, text_features, **args):
        # pre deal
        Us = [str(int(pair[0])) for pair in batch]
        Is = [str(int(pair[1])) for pair in batch]
        Js = [str(int(pair[2])) for pair in batch]
        Ks = [str(int(pair[3])) for pair in batch]
        # part one General
        if not self.visual_nn[0].weight.data.is_cuda:
            I_visual_latent = self.visual_nn(cat(
                [visual_features[I].unsqueeze(0) for I in Is], 0
            ))
            J_visual_latent = self.visual_nn(cat(
                [visual_features[J].unsqueeze(0) for J in Js], 0
            ))
            K_visual_latent = self.visual_nn(cat(
                [visual_features[K].unsqueeze(0) for K in Ks], 0
            ))
            I_text_latent = self.textcnn( 
                self.text_embedding( 
                    cat(
                        [text_features[I].unsqueeze(0) for I in Is], 0
                    )
                ) .unsqueeze_(1)
            )
            J_text_latent = self.textcnn( 
                self.text_embedding( 
                    cat(
                        [text_features[J].unsqueeze(0) for J in Js], 0
                    )
                ).unsqueeze_(1)
            )
            K_text_latent = self.textcnn( 
                self.text_embedding( 
                    cat(
                        [text_features[K].unsqueeze(0) for K in Ks], 0
                    )
                ) .unsqueeze_(1)
            )

        else :
            with torch.cuda.device(self.visual_nn[0].weight.data.get_device()):
                stream1 = torch.cuda.Stream()
                stream2 = torch.cuda.Stream()
                I_visual_latent = self.visual_nn(cat(
                    [visual_features[I].unsqueeze(0) for I in Is], 0
                ).cuda())
                with torch.cuda.stream(stream1):
                    J_visual_latent = self.visual_nn(cat(
                        [visual_features[J].unsqueeze(0) for J in Js], 0
                    ).cuda())
                with torch.cuda.stream(stream2):
                    K_visual_latent = self.visual_nn(cat(
                        [visual_features[K].unsqueeze(0) for K in Ks], 0
                    ).cuda())
                I_text_latent = self.textcnn( 
                    self.text_embedding( 
                        cat(
                            [text_features[I].unsqueeze(0) for I in Is], 0
                        ).cuda() 
                    ) .unsqueeze_(1)
                )
                with torch.cuda.stream(stream1):
                    J_text_latent = self.textcnn( 
                        self.text_embedding( 
                            cat(
                                [text_features[J].unsqueeze(0) for J in Js], 0
                            ) .cuda()
                        ).unsqueeze_(1)
                    )
                with torch.cuda.stream(stream2):
                    K_text_latent = self.textcnn( 
                        self.text_embedding( 
                            cat(
                                [text_features[K].unsqueeze(0) for K in Ks], 0
                            ) .cuda()
                        ) .unsqueeze_(1)
                    )

        visual_ij = bmm( I_visual_latent.unsqueeze(1), J_visual_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        print('visualij done')
        text_ij = bmm( I_text_latent.unsqueeze(1), J_text_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        print('textij done')
        cuj = self.vtbpr(Us, Js, J_visual_latent, J_text_latent)
        print('cuj done')
        visual_ik = bmm( I_visual_latent.unsqueeze(1), K_visual_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        print('visualik done')
        text_ik = bmm( I_text_latent .unsqueeze(1), K_text_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        print('textik done')
        cuk = self.vtbpr(Us, Ks, K_visual_latent, K_text_latent)
        print('cuk done')
        
        # # part 2 cuj
        # torch.cuda.synchronize()
        # stream1 = torch.cuda.Stream()
        # stream2 = torch.cuda.Stream()
        # visual_ij = bmm( I_visual_latent.unsqueeze(1), J_visual_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        # with torch.cuda.stream(stream1):
        #     text_ij = bmm( I_text_latent.unsqueeze(1), J_text_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        #     cuj = self.vtbpr(Us, Js, J_visual_latent, J_text_latent)
        # visual_ik = bmm( I_visual_latent.unsqueeze(1), K_visual_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        # with torch.cuda.stream(stream2):
        #     text_ik = bmm( I_text_latent .unsqueeze(1), K_text_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        #     cuk = self.vtbpr(Us, Ks, K_visual_latent, K_text_latent)
        
        # torch.cuda.synchronize()
        
        p_ij = (1 - 0.5) * visual_ij + 0.5 * text_ij
        p_ik = (1 - 0.5) * visual_ik + 0.5 * text_ik
        # union
        return self.uniform_value * p_ij + (1 - self.uniform_value) * cuj \
            - ( self.uniform_value * p_ik + (1 - self.uniform_value) * cuk )
    def fit(self, batch, visual_features, text_features, **args):
        """
            with the same input as forward and return a loss with weight regularaition
        """

        Us = [str(int(pair[0])) for pair in batch]
        Is = [str(int(pair[1])) for pair in batch]
        Js = [str(int(pair[2])) for pair in batch]
        Ks = [str(int(pair[3])) for pair in batch]
        if not self.visual_nn[0].weight.data.is_cuda:
            I_visual_latent = self.visual_nn(cat(
                [visual_features[I].unsqueeze(0) for I in Is], 0
            ))
            J_visual_latent = self.visual_nn(cat(
                [visual_features[J].unsqueeze(0) for J in Js], 0
            ))
            K_visual_latent = self.visual_nn(cat(
                [visual_features[K].unsqueeze(0) for K in Ks], 0
            ))
            I_text_latent = self.textcnn( 
                self.text_embedding( 
                    cat(
                        [text_features[I].unsqueeze(0) for I in Is], 0
                    )
                ) .unsqueeze_(1)
            )
            J_text_latent = self.textcnn( 
                self.text_embedding( 
                    cat(
                        [text_features[J].unsqueeze(0) for J in Js], 0
                    )
                ).unsqueeze_(1)
            )
            K_text_latent = self.textcnn( 
                self.text_embedding( 
                    cat(
                        [text_features[K].unsqueeze(0) for K in Ks], 0
                    )
                ) .unsqueeze_(1)
            )

        else :
            with torch.cuda.device(self.visual_nn[0].weight.data.get_device()):
                stream1 = torch.cuda.Stream()
                stream2 = torch.cuda.Stream()
                I_visual_latent = self.visual_nn(cat(
                    [visual_features[I].unsqueeze(0) for I in Is], 0
                ).cuda())
                with torch.cuda.stream(stream1):
                    J_visual_latent = self.visual_nn(cat(
                        [visual_features[J].unsqueeze(0) for J in Js], 0
                    ).cuda())
                with torch.cuda.stream(stream2):
                    K_visual_latent = self.visual_nn(cat(
                        [visual_features[K].unsqueeze(0) for K in Ks], 0
                    ).cuda())
                I_text_latent = self.textcnn( 
                    self.text_embedding( 
                        cat(
                            [text_features[I].unsqueeze(0) for I in Is], 0
                        ).cuda() 
                    ) .unsqueeze_(1)
                )
                with torch.cuda.stream(stream1):
                    J_text_latent = self.textcnn( 
                        self.text_embedding( 
                            cat(
                                [text_features[J].unsqueeze(0) for J in Js], 0
                            ) .cuda()
                        ).unsqueeze_(1)
                    )
                with torch.cuda.stream(stream2):
                    K_text_latent = self.textcnn( 
                        self.text_embedding( 
                            cat(
                                [text_features[K].unsqueeze(0) for K in Ks], 0
                            ) .cuda()
                        ) .unsqueeze_(1)
                    )
        # part 2 cuj
        torch.cuda.synchronize()
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        visual_ij = bmm( I_visual_latent.unsqueeze(1), J_visual_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        with torch.cuda.stream(stream1):
            text_ij = bmm( I_text_latent.unsqueeze(1), J_text_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
            cuj, cujweight = self.vtbpr.fit(Us, Js, J_visual_latent, J_text_latent)
        visual_ik = bmm( I_visual_latent.unsqueeze(1), K_visual_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
        with torch.cuda.stream(stream2):
            text_ik = bmm( I_text_latent .unsqueeze(1), K_text_latent .unsqueeze(-1)).squeeze_(-1).squeeze_(-1)
            cuk, cukweight = self.vtbpr.fit(Us, Ks, K_visual_latent, K_text_latent)
        
        torch.cuda.synchronize()
        
        p_ij = (1 - 0.5) * visual_ij + 0.5 * text_ij
        p_ik = (1 - 0.5) * visual_ik + 0.5 * text_ik
        
        cujkweight = self.vtbpr.get_user_gama(set(Us)).norm(p=2) \
            + self.vtbpr.get_theta_user_visual(set(Us)).norm(p=2) + self.vtbpr.get_theta_user_text(set(Us)).norm(p=2) \
            + self.vtbpr.get_item_gama(set(Js+Ks)).norm(p=2)
        
        
        # union
        return self.uniform_value * p_ij + (1 - self.uniform_value) * cuj \
            - ( self.uniform_value * p_ik + (1 - self.uniform_value) * cuk ) ,\
                cujkweight + self.text_embedding( 
                            cat(
                                [text_features[J].unsqueeze(0) for J in set(Is+Js+Ks)], 0
                            ) .cuda()
                        ).norm(p=2)
      