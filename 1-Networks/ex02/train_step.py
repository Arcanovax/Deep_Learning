def one_step(model, x, y, loss_fn, optimizer):
    ouput = model(x)
    loss = loss_fn(ouput, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
