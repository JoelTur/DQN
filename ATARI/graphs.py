import neptune.new as neptune

def graph(rewards, avg_rewards, projectname):


    run = neptune.init(
    project=projectname,
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTIzZmNmNS04YjRiLTQyMWMtYmIzMy1kOGEwNTE0NmRjOWQifQ==",
)  # your credentials

    for i  in range(len(rewards)):
        run["rewards per episode"].log(rewards[i])
        run["avg reward per episode"].log(avg_rewards[i])



    run.stop()