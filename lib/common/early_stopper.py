class EarlyStopper(object):
    def __init__(self, patience, saveable):
        self.patience = patience
        self.saveable = saveable
        self.best_val_loss = None
        self.best_info = None
        self.early_stopping_count = 0

    def register_loss(self, loss):
        if self.best_val_loss is None:
            self.best_val_loss = loss
            self.saveable.best_info = self.saveable.get_info()
            self.early_stopping_count = 0
        elif self.best_val_loss > loss:
            self.best_val_loss = loss
            self.saveable.best_info = self.saveable.get_info()
            self.early_stopping_count = 0
        else:
            self.early_stopping_count += 1

    def should_stop(self):
        return self.early_stopping_count == self.patience
