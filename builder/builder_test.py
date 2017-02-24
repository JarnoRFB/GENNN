from builder.network_builder import Network

with open('../test_spec_simple.json') as fp:
    spec = str(fp.read())
    # print(spec)
    network = Network(spec)

    print(network.evaluate())