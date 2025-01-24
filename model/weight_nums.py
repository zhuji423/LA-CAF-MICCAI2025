weight_nums.append(8*8)
weight_nums.append(8*8)
weight_nums.append(8*1)  ##weight_nums[64,64,8]
bias_nums.append(8)
bias_nums.append(8)
bias_nums.append(1) ## bias_nums[8,8,1]
if self.is_concat:
## solution 1 : concat
    x_cond = torch.cat([x_attention_feat, task_encoding], 1) ## 都变化为[34,256 -->512,1,1,1] 两个按照第一个维度拼接
    params = self.controller(x_cond) ## torch.Size([34, 153, 1, 1, 1]) 经过了一个controller 卷积
else:
## solution 2 : no concat
    x_cond = x_attention_feat
    params = self.controller_256(x_cond)

params.squeeze_(-1).squeeze_(-1).squeeze_(-1) ##[34, 153]
head_inputs = self.precls_conv(out[i].unsqueeze(0)) ## input:[1,48,96,96,96] output:[1,8,96,96,96]
head_inputs = head_inputs.repeat(self.class_num,1,1,1,1)  ## [34,8,96,96,96]
N, _, D, H, W = head_inputs.size()
head_inputs = head_inputs.reshape(1, -1, D, H, W) ##[1,272,96,96,96]
# print(head_inputs.shape, params.shape)
weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)
logits = self.heads_forward(head_inputs, weights, biases, N)   ## 将参数作为卷积核的参数得到最终的结果 [1,34,96,96,96]

def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2 ## torch.Size([34, 153]) channels = 8
        assert len(weight_nums) == len(bias_nums) ## weight_nums[64,64,8]
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1  ## 列表的拼接：[64,64,8,8,8,1] 按照第1维进行切分，每一个向量都是[34,x],x是前面的列表
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1) ## [272,8,1,1,1]
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x