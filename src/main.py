from model.model import Model
import numpy as np

import tensorflow as tf

from server.app import App

if __name__ == "__main__":


    model = Model(784, 64, 28)
    app = App(model)

    model.train()
    print("ready")

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)



