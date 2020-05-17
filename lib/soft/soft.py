import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, hidden_size, is_coord):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3 if not is_coord else 5,
            out_channels=64,
            kernel_size=5,
            stride=2
        )
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2)
        self.batch_norm2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=hidden_size, kernel_size=5, stride=2)
        self.batch_norm3 = torch.nn.BatchNorm2d(hidden_size)
        self.activ = nn.ReLU()

    def forward(self, x):
        out_conv1 = self.activ(self.batch_norm1(self.conv1(x)))
        out_conv2 = self.activ(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = self.activ(self.batch_norm3(self.conv3(out_conv2)))
        return out_conv3


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=6)
        )

    def forward(self, x):
        return self.model(x)


class SoftAttention(nn.Module):
    def __init__(self, hidden_size, projection_scale=1):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.fc_combine = nn.Linear(in_features=hidden_size // projection_scale, out_features=1)
        self.fc_v_t = nn.Linear(in_features=hidden_size, out_features=hidden_size // projection_scale)
        self.fc_h_t = nn.Linear(in_features=hidden_size, out_features=hidden_size // projection_scale)
        self.activ = nn.ReLU()

    def forward(self, x, hidden_state):
        # x (batch, C', H'xW')
        b, c_p, h_pxw_p = x.size()
        # h_t (batch, C' = hidden_size)
        if hidden_state is None:
            h_t = torch.zeros((b, c_p)).to(x.device)
        else:
            h_t, _ = hidden_state
            # remove seq len dimension
            h_t = h_t.squeeze(0)

        # M = C' // 4 = hidden_size // projection_scale
        # proj_h_t (batch, M)
        proj_h_t = self.fc_h_t(h_t)
        # proj_vs (batch*H'*W', M)
        proj_vs = self.fc_v_t(x.transpose(1, 2).contiguous().view(b * h_pxw_p, c_p))
        # Linear(v_t^i) + Linear(h_t)
        # (batch, H'*W', M) + (batch, 1, M)
        sum_projs = proj_vs.view(b, h_pxw_p, -1) + proj_h_t.unsqueeze(1)
        # sum_projs (batch, H'*W', M)
        result = self.fc_combine(self.activ(sum_projs.view(b * h_pxw_p, -1)))
        # result (batch*H'*W', 1)
        result = result.view(b, h_pxw_p, 1).transpose(1, 2)
        importance = self.softmax(result)
        # out (batch, 1, H'xW')
        return importance


class SoftCNNLSTMNetwork(nn.Module):
    def __init__(self, hidden_size, is_coord, projection_scale):
        super().__init__()
        self.cnn = CNN(hidden_size=hidden_size, is_coord=is_coord)
        self.attention = SoftAttention(hidden_size=hidden_size, projection_scale=projection_scale)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.mlp = MLP(hidden_size=hidden_size)
        self.importance = None
        self.h_p = None
        self.conv1_up = nn.UpsamplingBilinear2d(size=(96, 128))

    def get_upsampled_attention(self):
        b, _1, h_pxw_p = self.importance.size()
        importance = self.importance.view(b, 1, self.h_p, -1)
        return self.conv1_up(importance)

    def forward(self, x, hidden_state=None):
        # (batch, d_len, C, H, W)
        b, d_len, c, h, w = x.size()
        x = x.view(b * d_len, c, h, w)
        out_cnn = self.cnn(x)
        bxd_len, c_p, h_p, w_p = out_cnn.size()
        self.h_p = h_p
        out_cnn = out_cnn.view(b, d_len, c_p, h_p, w_p)
        # output visual features: (batch, demonstration_length, C', H', W')
        out_cnn_view = out_cnn.transpose(0, 1)
        out_cnn_feature_vectors = out_cnn_view.view(d_len, b, c_p, h_p * w_p)
        # apply soft attention, inputs
        # 1: (demonstration_length, batch, C', H'*W')
        # 2: current hidden state (demonstration_length, batch, hidden_size)
        out = torch.zeros((d_len, b, 6)).to(x.device)
        importances = []
        for d_step, batch in enumerate(out_cnn_feature_vectors):
            # batch: (batch, C', H'xW')
            assert batch.size() == (b, c_p, h_p * w_p)
            importance = self.attention(batch, hidden_state)
            # store current importance for easy access
            self.importance = importance
            importances.append(importance)
            # importance (batch, 1, H'xW')
            assert importance.size() == (b, 1, h_p * w_p)
            # context variable (batch, hidden_size)
            z_t = torch.sum(importance * batch, dim=-1)
            assert z_t.size() == (b, c_p)
            # unsqueeze to add sequence dimension
            output, hidden_state = self.lstm(z_t.unsqueeze(0), hx=hidden_state)
            out[d_step] = self.mlp(output.squeeze(0))

        # so out is (b, seq_len, 6) similarly to input
        return out.transpose(0, 1), hidden_state, importances
