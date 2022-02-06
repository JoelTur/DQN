import wandb

def graph(rewards, avg_rewards, loss, projectname):


    wandb.init(project="BREAKOUT_DQN", entity="neuroori") 
    data = [[x,y] for x, y in enumerate(rewards)]
    table = wandb.Table(data= data, columns=["Episode", "Reward"])
    wandb.config = {"learning_rate": 0.00025, "batch_size": 32}
    wandb.log({"Reward per episode": wandb.plot.line(table, "Episode", "Reward", title = "Reward per episode")})
    data = [[x,y] for x, y in enumerate(avg_rewards)]
    table = wandb.Table(data= data, columns=["Episode", "Avg_Reward"])
    wandb.log({"Avg reward": wandb.plot.line(table, "Episode", "Avg_Reward", title = "Avg reward")})
    data = [[x,y] for x, y in enumerate(loss)]
    table = wandb.Table(data= data, columns=["Episode", "Loss"])
    wandb.log({"Loss": wandb.plot.line(table, "Episode", "Loss", title = "Loss")})