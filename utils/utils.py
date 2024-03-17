import numpy as np
import pandas as pd
import torch
import umap
from torch.utils.data import DataLoader
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


def prepare_batch(cifar_data_module):
    dataloader = DataLoader(cifar_data_module.test_data, batch_size=cifar_data_module.batch_size, shuffle=True,  num_workers=cifar_data_module.num_workers)
    num_probes_of_class = 5
    classes_id = cifar_data_module.classes_to_learn + cifar_data_module.classes_to_dream
    dataiter = iter(dataloader)
    batch_img, batch_targets = next(dataiter)
    batch_img_shape = batch_img.shape
    batch_target_shape = batch_targets.shape
    prep_imgs = [torch.empty((0, *batch_img_shape[1:])) for _ in range(len(classes_id))]
    prep_targets = [torch.empty((0)) for _ in range(len(classes_id))]
    counted = 0
    while num_probes_of_class * len(classes_id) > counted:
        batch_img, batch_targets = next(iter(dataloader))

        for idx, class_id in enumerate(classes_id):
            looking_num_probes = num_probes_of_class - len(prep_targets[idx])
            if looking_num_probes > 0:
                mask = batch_targets == class_id
                indices = torch.nonzero(mask)[:looking_num_probes]
                prep_imgs[idx].cat(batch_img[indices], 0)
                prep_targets[idx].cat(batch_targets[indices], 0)
                counted += len(indices)

    prep_imgs = torch.cat(prep_imgs, axis=0)
    prep_targets = torch.cat(prep_imgs, axis=0)

    return prep_imgs, prep_targets

def visualize_output_space(mlf_logger, images, embeddings, labels, step="train", num_classes = 5, example_size = 10):
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
    )

    for location, example in zip(examples_locations, examples):
        x, y = location[0], location[1]
        label = int(location["label"])

        ab = AnnotationBbox(OffsetImage(example, zoom=1), (x, y), frameon=True,
                            bboxprops=dict(facecolor=sns.color_palette("bright", num_classes)[label], boxstyle="round"))
        ax.add_artist(ab)
    img_path = os.path.join(os.getcwd(), 'trained', f'output_space_{step}.png')
    plt.savefig(img_path)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, img_path)
