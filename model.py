from torch import nn

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.nn = nn.Sequential(
            # fc4
            nn.Linear(512*2*2, 512),
            nn.ReLU(),
            nn.Dropout(),

            # fc5
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),

            # output
            nn.Linear(512, 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, .1)
                m.bias.data.zero_()

    def forward(self, feat):
        feat = feat.view(feat.size(0), -1)
        return self.nn(feat)
