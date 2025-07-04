

def run(config, manager, exporter):

    for epoch in range(config["algorithm_epochs"]):

        manager.predict()
        manager.train()
        manager.drop(epoch)
        manager.add(epoch)
        exporter.export_data(epoch)

    manager.predict()
