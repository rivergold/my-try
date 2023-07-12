import torch
import sympy as sp

if __name__ == '__main__':
    model = torch.nn.Conv2d(1, 1, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[30, 80],
                                                      gamma=0.1)

    for epoch in range(50):
        # for input, target in dataset:
        #     optimizer.zero_grad()
        #     output = model(input)
        #     loss = loss_fn(output, target)
        #     loss.backward()
        #     optimizer.step()
        scheduler1.step()
        scheduler2.step()

        print(epoch, optimizer.param_groups[-1]['lr'])
