from utils.networks import *
from utils.preprocess import *
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
import tensorflow as tf

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# Source: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html


def select_n_random(data, labels, n=100):
    """Selects n random datapoints and their corresponding labels from a dataset"""
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def images_to_probs(net, images):
    """Generates predictions & corresponding probabilities from a trained network and a list of iamges"""
    output = net(images)

    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig


def train(net, criterion, optimizer, tb_writer):
    running_loss = 0.0
    for epoch in range(1):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:  # every 1000 mini-batches...

                # ...log the running loss
                tb_writer.add_scalar(
                    "training loss", running_loss / 1000, epoch * len(trainloader) + i
                )

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                tb_writer.add_figure(
                    "predictions vs. actuals",
                    plot_classes_preds(net, inputs, labels),
                    global_step=epoch * len(trainloader) + i,
                )
                running_loss = 0.0
    print("Finished Training")

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_label, writer, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

def main():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # create a summary writer that store the local host of TensorBoard
    writer = SummaryWriter("runs/fm_exp_1")

    # write to TensorBoard with grid - make_grid
    # get randome training iamges and corresponding labels
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img=img_grid, one_channel=True)

    # write to Tensorboard -------------------------------------------
    writer.add_image("4_fm_images", img_grid)

    # inspect the model
    writer.add_graph(net, images)
    # writer.close()

    # select random images and their target indices # Tensorboard-----
    images, labels = select_n_random(trainset.data, trainset.targets)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    # Projector show 3D of 100 images with each is 784 dimensional projected down into 3D space/
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
    writer.close()

    # Tracking model training with TensorBoard:
    # Log the running loss to TensorBoard, along with a view into the predictions the model is making via plot_classes_preds function
    train(net=net, criterion=criterion, optimizer=optimizer, tb_writer=writer)

    # Assess trained model: 
    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_label = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(labels)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_label, writer)

if __name__ == "__main__":
    print("Running main()")
    main()
