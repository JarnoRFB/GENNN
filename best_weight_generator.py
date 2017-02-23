import json
import os
from builder.network_builder import Network

class Generator:

    def __init__(self):
        self.best_accuracy = 0
        self.best_accuracy_logdir = None
        self.best_network_spec = None

    def get_best_spec (self, logdir):
        # get all netswork.json
        for subdir, dirs, files in os.walk(str(logdir)):
            for file in files:
                # filter
                if file == 'network.json':
                    file_loc = os.path.join(subdir, file)
                    #read current network.json
                    with open(file_loc, 'r') as fp:
                        n_spec = json.loads( fp.read() )
                        if('results' in n_spec ):
                            if(self.best_accuracy < n_spec['results']['accuracy']):
                                self.best_accuracy = n_spec['results']['accuracy']
                                self.best_accuracy_logdir = file_loc
                                self.best_network_spec = n_spec

        #self._write_to_logdir(logdir=logdir, n_spec=self.best_network_spec)
        return self.best_network_spec

    def _write_to_logdir(self, logdir, n_spec):
        file_loc = os.path.join(logdir, 'Best_Network.json')
        with open(file_loc, 'w') as fp:
            fp.write(str(n_spec))




def main():
            base_logdir = 'log'
            output_logdir = base_logdir + '/best/'

            generator = Generator()
            json_network_spec = generator.get_best_spec(logdir = base_logdir)
            print('best Network: ' + str(json_network_spec))

            # workaround
            json_network_spec['logdir'] = output_logdir

            network = Network(json.dumps(json_network_spec))
            network.evaluate(get_weights=True)

            print('Finished')

main()