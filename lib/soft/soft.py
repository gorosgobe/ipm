import numpy as np
import torch
import torch.nn.functional as F
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
        self.hidden_size = hidden_size
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

    def get_input_size_for(self, size):
        h, w = size
        h_p = (h - 5) // 2 + 1
        h_p = (h_p - 7) // 2 + 1
        h_p = (h_p - 5) // 2 + 1
        w_p = (w - 5) // 2 + 1
        w_p = (w_p - 7) // 2 + 1
        w_p = (w_p - 5) // 2 + 1
        return self.hidden_size, h_p, w_p


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=6),
        )

    def forward(self, x):
        return self.model(x)


class SoftmaxProbActiv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.activ = nn.Softmax(dim=dim)

    def forward(self, x, _hidden_state):
        return self.activ(x)


class GumbelSigmoidProbActiv(nn.Module):
    def __init__(self, hidden_size, adaptive_tau, annealed_tau):
        super().__init__()
        if adaptive_tau + annealed_tau != 1:
            raise ValueError("Either annealed tau or learnt tau are required!")

        self.adaptive_tau = adaptive_tau
        if self.adaptive_tau:
            self.tau_layer = nn.Linear(in_features=hidden_size, out_features=1)
            self.softplus = nn.Softplus()

        # otherwise, annealing parameters
        self.annealed_tau = annealed_tau
        # initial value for tau
        self.max_tau = self.tau = 1.0
        self.iter = 0
        self.annealing_rate = 0.0002
        self.min_tau = 0.3

        # elementwise sigmoid
        self.activ = nn.Sigmoid()

    def forward(self, x, hidden_state):
        # x (batch, 1, H'xW')
        # hidden_state (b, hidden_size)
        if self.annealed_tau:
            self.iter += 1
            # annealing as in https://arxiv.org/pdf/1805.02336.pdf
            # anneal tau
            self.tau = max(self.min_tau, self.max_tau * np.exp(-self.annealing_rate * self.iter))
        else:
            # adaptive, learnt tau
            # adaptive tau mechanism from https://arxiv.org/abs/1701.08718 and https://arxiv.org/pdf/1708.07590.pdf
            self.tau = 1 / (1 + self.softplus(self.tau_layer(hidden_state)))
            b, _ = hidden_state.size()
            self.tau = self.tau.view(b, 1, 1, 1)

        sigmoid_x = self.activ(x.unsqueeze(-1))
        sigmoid_x = torch.clamp(sigmoid_x, 1e-7, 1 - 1e-7)
        # sigmoid_x (batch, 1, H'xW', 1)
        # given tensor with probabilities p for feature vector i computed from an element-wise sigmoid
        # calculate 1 - p, zip to obtain tensor with [p, 1 - p], apply log
        sigmoid_both_log_probs = torch.log(torch.cat((sigmoid_x, 1 - sigmoid_x), dim=-1))
        # sigmoid_both_log_probs (batch, 1, H'xW', 2)
        # then apply gumbel softmax and pick p's index as part of the mask
        gumbel_out = nn.functional.gumbel_softmax(sigmoid_both_log_probs, tau=self.tau, hard=True)
        indexed_gumbel_out = gumbel_out[:, :, :, :1].squeeze(-1)
        return indexed_gumbel_out


class DummySigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, _hidden_state):
        return self.sigmoid(x)


class SoftAttention(nn.Module):
    def __init__(self, hidden_size, is_mask=False, gumbel_params=None, projection_scale=1):
        super().__init__()
        if gumbel_params is None:
            self.prob_activ = SoftmaxProbActiv(dim=-1) if not is_mask else DummySigmoid()
        else:
            self.prob_activ = GumbelSigmoidProbActiv(
                hidden_size=hidden_size,
                adaptive_tau=gumbel_params["adaptive_tau"],
                annealed_tau=gumbel_params["annealed_tau"]
            )
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
        # result (batch, 1, H'xW')
        importance = self.prob_activ(result, h_t)
        # out (batch, 1, H'xW')
        return importance


class SoftCNNLSTMNetwork(nn.Module):
    def __init__(self, hidden_size, is_coord, projection_scale, keep_masked=False, gumbel_params=None):
        super().__init__()
        self.keep_masked = keep_masked
        self.cnn = CNN(hidden_size=hidden_size, is_coord=is_coord)
        self.attention = SoftAttention(
            hidden_size=hidden_size,
            is_mask=keep_masked,
            projection_scale=projection_scale,
            gumbel_params=gumbel_params
        )
        if self.keep_masked:
            v_input_size = int(np.product(self.cnn.get_input_size_for((96, 128))))
            # takes full size of image features
            self.lstm = nn.LSTM(input_size=v_input_size, hidden_size=hidden_size)
            self.mlp = MLP(input_size=v_input_size)
        else:
            self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
            self.mlp = MLP(input_size=hidden_size)
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
            if self.keep_masked:
                z_t = importance * batch
                z_t = z_t.view(b, c_p * h_p * w_p)
            else:
                z_t = torch.sum(importance * batch, dim=-1)
                assert z_t.size() == (b, c_p)
            # unsqueeze to add sequence dimension
            output, hidden_state = self.lstm(z_t.unsqueeze(0), hx=hidden_state)
            out[d_step] = self.mlp(z_t)

        # so out is (b, seq_len, 6) similarly to input
        return out.transpose(0, 1), hidden_state, importances


class RecurrentBaseline(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size)
        self.fc1 = torch.nn.Linear(in_features=hidden_size + 6, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x, hidden_state=None):
        # x (b, d_len, 6)
        out_lstm, _hidden_state = self.lstm(x.transpose(0, 1))
        # out_lstm (d_len, b, hidden_size)
        transposed_out_lstm = out_lstm.transpose(0, 1)
        out_lstm_concat_rels = torch.cat((transposed_out_lstm, x), dim=-1)
        out_fc1 = F.relu(self.fc1.forward(out_lstm_concat_rels))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3, None, None


class RecurrentFullImage(nn.Module):
    def __init__(self, hidden_size, is_coord, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv1 = torch.nn.Conv2d(in_channels=3 if not is_coord else 5, out_channels=64, kernel_size=5, stride=2,
                                     padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        flattened_size = 2240
        self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=hidden_size)
        self.fc1 = torch.nn.Linear(in_features=hidden_size + flattened_size, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x, hidden_state=None):
        # x(batch, d_len, C, H, W)
        b, d_len, c, h, w = x.size()
        x = x.view(b * d_len, c, h, w)
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(x)))
        out_conv2 = F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(b, d_len, -1)
        # out_conv3 (b, d_len, c * h * w)
        out_lstm, hidden_state = self.lstm(out_conv3.transpose(0, 1))
        # out_lstm (d_len, b, hidden)
        out_lstm = out_lstm.transpose(0, 1)
        out_lstm_concat_visual = torch.cat((out_lstm, out_conv3), dim=-1)
        out_fc1 = F.relu(self.fc1.forward(out_lstm_concat_visual))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        # out is (b, d_len, 6)
        return out_fc3, None, None


class RecurrentCoordConv_32(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        # spatial information is encoded as coord feature maps, one for x and one for y dimensions, fourth/fifth channels
        self.conv1 = torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        flattened_size = 32
        self.lstm = nn.LSTM(input_size=flattened_size, hidden_size=hidden_size)
        self.fc1 = torch.nn.Linear(in_features=hidden_size + flattened_size, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, image_batch, hidden_state=None):
        b, d_len, c, h, w = image_batch.size()
        image_batch = image_batch.view(b * d_len, c, h, w)
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(image_batch)))
        out_conv2 = F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(b, d_len, -1)
        # out_conv3 (b, d_len, c * h * w)
        out_lstm, hidden_state = self.lstm(out_conv3.transpose(0, 1))
        # out_lstm (d_len, b, hidden)
        out_lstm = out_lstm.transpose(0, 1)
        # TODO: concat flattened outconv3 and LSTM output, and use that instead.
        out_lstm_concat_visual = torch.cat((out_lstm, out_conv3), dim=-1)
        out_fc1 = F.relu(self.fc1.forward(out_lstm_concat_visual))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        # out is (b, d_len, 6)
        return out_fc3, None, None
