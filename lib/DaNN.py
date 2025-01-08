import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.Dropout(),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            # nn.Linear(512, 512),
            # nn.Dropout(),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            
            nn.Linear(512, 512),
        )
        
    def forward(self, x):
        # x = F.softmax(self.fc(x), dim=1)
        x = self.fc(x)
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )

    def forward(self, h):
        c = self.layer(h)
        c = F.softmax(c, dim=1)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 6),
        )

    def forward(self, h):
        y = self.layer(h)
        return y