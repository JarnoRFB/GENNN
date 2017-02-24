from builder.network_builder import Network

with open('../acc_test.json') as fp:
    spec = str(fp.read())
    # print(spec)
    network = Network(spec)

    print(network.evaluate())