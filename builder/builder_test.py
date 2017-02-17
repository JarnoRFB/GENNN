from builder.network_builder import Network

with open('../test_spec.json') as fp:
    spec = str(fp.read())
    # print(spec)
    network = Network(spec)

    print(network.evaluate())