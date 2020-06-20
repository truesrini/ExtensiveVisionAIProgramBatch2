<<<<<<< HEAD
import torch

def train_test_model (model, device, trainloader, testloader, optimizer, criterion, epochs, model_name):
    print("TRAINING STARTS")
    train_accuracy_list = []
    test_accuracy_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model.to(device)
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print("********** EPOCH NUMBER IS ",epoch," *************") 
        print('Training Accuracy: %d %%' % (100 * correct / total))
        train_accuracy_list.append(100 * correct / total)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            test_accuracy = (100 * correct / total)
            test_accuracy_list.append(test_accuracy)
            if test_accuracy >= max(test_accuracy_list):
                torch.save(model.state_dict(),model_name)

        print('Test Accuracy: %d %%' % (100 * correct / total))


    print('Finished Training')
=======
import torch

def train_test_model (model, device, trainloader, testloader, optimizer, criterion, epochs, model_name):
    print("TRAINING STARTS")
    train_accuracy_list = []
    test_accuracy_list = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model.to(device)
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print("********** EPOCH NUMBER IS ",epoch," *************") 
        print('Training Accuracy: %d %%' % (100 * correct / total))
        train_accuracy_list.append(100 * correct / total)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            test_accuracy = (100 * correct / total)
            test_accuracy_list.append(test_accuracy)
            if test_accuracy >= max(test_accuracy_list):
                torch.save(model.state_dict(),model_name)

        print('Test Accuracy: %d %%' % (100 * correct / total))


    print('Finished Training')
>>>>>>> 5d248e4e6ce69c748e354d2362a1cfeabcc61bfb
    return train_accuracy_list, test_accuracy_list