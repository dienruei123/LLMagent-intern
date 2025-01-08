import torch.nn as nn
import torch.nn.functional as F

class TrueFalseClassifier(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.fc = nn.Sequential(
    #         nn.Linear(4096, 256),
    #         # nn.Dropout(),
    #         nn.BatchNorm1d(256),
    #         nn.ReLU(),
            
    #         nn.Linear(256, 128),
    #         # nn.Dropout(),
    #         nn.BatchNorm1d(128),
    #         nn.ReLU(),
            
    #         nn.Linear(128, 64),
    #         nn.Linear(64, 2),
    #     )

    # def forward(self, x):
    #     x = F.softmax(self.fc(x), dim=1)
    #     return x
    
    # def __init__(self):
    #     super().__init__()
    #     self.fc1 = nn.Linear(4096, 256)
    #     self.fc3 = nn.Linear(256, 128)
    #     self.fc4 = nn.Linear(128, 64)
    #     self.fc5 = nn.Linear(64, 2)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     # x = F.relu(self.fc2(x))
    #     x = F.relu(self.fc3(x))
    #     x = F.relu(self.fc4(x))
    #     x = F.softmax(self.fc5(x), dim=1)
    #     return x
    
    
    # match the same size as DaNN
    def __init__(self):
        super().__init__()
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
            
            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x