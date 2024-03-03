import numpy as np
import pandas as pd
import torch
import umap
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import seaborn as sns
import os
umap = umap.UMAP(metric="cosine", n_neighbors=100)

def embed_imgs(model, batch):
    img_list, embed_list = [], []
    inputs, targets = batch
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    labels = []
    inputs = inputs.to(device)
    model.to(device)
    with torch.no_grad():
        preds = model(inputs)
    img_list.append(inputs.cpu())
    embed_list.append(preds.cpu())
    labels.append(targets)
    return (torch.cat(img_list, dim=0), torch.cat(embed_list, dim=0), torch.cat(labels, dim=0))


def visualize_output_space(mlf_logger, images, embeddings, labels, step="train", example_size = 10):
    train_embedded = umap.fit_transform(embeddings)
    data = pd.DataFrame(train_embedded)
    data["label"] = labels
    examples = []
    examples_locations = []
    for i in np.random.randint(0, len(images), example_size):
        img = images[i]
        img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2) / 2 + 0.5
        examples.append(img)
        examples_locations.append(data.iloc[i])

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(
        x=0, y=1,
        hue="label",
        palette=sns.color_palette("hls", 10),
        data=data,
        legend="full",
        alpha=0.1
    )

    for location, example in zip(examples_locations, examples):
        x, y = location[0], location[1]
        label = int(location["label"])
        ab = AnnotationBbox(OffsetImage(example, zoom=1), (x, y), frameon=True,
                            bboxprops=dict(facecolor=sns.color_palette("hls", 10)[label], boxstyle="round"))
        ax.add_artist(ab)
    img_path = os.path.join(os.getcwd(), 'trained', f'output_space_{step}.png')
    plt.savefig(img_path)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, img_path)
