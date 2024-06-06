model = GoogLeNet(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_model(model, train_loader, criterion, optimizer, num_epochs=25)

evaluate_model(model, test_loader, criterion)
