# Fine tuning code from https://github.com/openai/CLIP/issues/83

# Changes by capjamesg:
# - Allow loading classification folder dataset, use supervision for data loading
# - Add tqdm progress bar
# - Set batch size and epochs

import tqdm

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import clip
import supervision as sv

from torch.utils.data import Dataset, DataLoader

EPOCH = 7
BATCH_SIZE = 16

cd = sv.ClassificationDataset.from_folder_structure("images/")

classes = cd.classes

images = cd.images.keys()

device = (
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # If using GPU then use mixed precision training.
model, preprocess = clip.load(
    "ViT-B/32", device=device, jit=False
)  # Must set jit=False for training

image = preprocess(Image.open("valid/midnights.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(
    [
        "1989",
        "evermore",
        "fearless",
        "folklore",
        "lover",
        "midnights",
        "red taylors version",
        "reputation",
        "speak now",
        "speak now taylors version",
        "taylor swift title",
    ]
).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print max probs label
    probs = [round(float(p), 4) for p in probs[0]]

    max_prob = max(probs)
    max_prob_index = probs.index(max_prob)

    print(classes[max_prob_index])
    print(probs)


class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.title = clip.tokenize(
            list_txt
        )  # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        title = self.title[idx]
        return image, title


# use your own data
list_image_path = list(images)
list_txt = [name.split("/")[1] for name in images]
dataset = image_title_dataset(list_image_path, list_txt)
train_dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE
)  # Define your own dataloader


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(
        model
    )  # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in tqdm.tqdm(range(EPOCH)):
    for batch in train_dataloader:
        optimizer.zero_grad()

        images, texts = batch

        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (
            loss_img(logits_per_image, ground_truth)
            + loss_txt(logits_per_text, ground_truth)
        ) / 2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

image = preprocess(Image.open("valid/midnights.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(
    [
        "1989",
        "evermore",
        "fearless",
        "folklore",
        "lover",
        "midnights",
        "red taylors version",
        "reputation",
        "speak now",
        "speak now taylors version",
        "taylor swift title",
    ]
).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print max probs label
    probs = [round(float(p), 4) for p in probs[0]]

    max_prob = max(probs)
    max_prob_index = probs.index(max_prob)

    print(classes[max_prob_index])
    print(probs)
