from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


num_to_letter = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q',
                 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class ASLDataValidate(Dataset):
    def __init__(self, labels, list_IDs, transform=None):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(num_to_letter)*4

    def __getitem__(self, index):
        """Generates one sample of data"""
        letter = num_to_letter[index // 4]
        ID = self.list_IDs[index % 4] + 3000 * (index // 4) + 1
        X = Image.open('asl-alphabet/asl_alphabet_train/asl_alphabet_train/' + letter + '/' + letter + str(ID) + '.jpg')
        y = index // 4
        X = self.transform(X)
        return X, y


class ASLData(Dataset):
    def __init__(self, labels, list_IDs, transform=None):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        letter = num_to_letter[index // 3000]
        ID = self.list_IDs[index % 3000] + 1
        X = Image.open('asl-alphabet/asl_alphabet_train/asl_alphabet_train/' + letter + '/' + letter + str(ID) + '.jpg')
        y = index // 3000
        X = self.transform(X)
        return X, y


def create_data_set():
    transformZ = transforms.Compose([
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset = ASLData(num_to_letter, list(range(87000)), transform=transformZ)
    dataload = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    return dataload, len(dataset)


def create_validation_set():
    transformZ = transforms.Compose([
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset = ASLDataValidate(num_to_letter, list(range(len(num_to_letter)*4)), transform=transformZ)
    dataload = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    return dataload, len(dataset)
