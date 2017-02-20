from builder.network_builder import Network

with open('../test_ff.json') as fp:
    spec = str(fp.read())
    # print(spec)
    network = Network(spec)

    print(network.evaluate())