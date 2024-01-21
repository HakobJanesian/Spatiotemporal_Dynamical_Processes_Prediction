
class LOSS:
    def __init__(self, mse=None, msmse=None, rimse=None, fmse=None):
        self.data = pd.DataFrame({
            'mse': [mse],
            'msmse': [msmse],
            'rimse': [rimse],
            'fmse': [fmse]
        })

    def plot(self, title="Plotting MSE Values", subplot_columns=None):
        if subplot_columns is None:
            subplot_columns = [self.data.columns.tolist()]

        fig, axes = plt.subplots(len(subplot_columns), 1)
        if len(subplot_columns) == 1:
            axes = [axes]

        for i, cols in enumerate(subplot_columns):
            for col in cols:
                if '+' in col:  # For combined plots like 'rimse+msmse'
                    combined_cols = col.split('+')
                    self.data[combined_cols].sum(axis=1).plot(ax=axes[i], label=col)
                else:
                    self.data[col].plot(ax=axes[i], label=col)

            axes[i].set_title(title)
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def __add__(self, L):
        new_loss = LOSS()
        new_loss.data = pd.concat([self.data, L.data]).reset_index(drop=True)
        return new_loss

# Creating two new instances of the LOSS class with different values
loss1 = LOSS(mse=0.3, msmse=0.2, rimse=0.25, fmse=1)
loss2 = LOSS(mse=0.35, msmse=0.15, rimse=0.22, fmse=2)

# Displaying the data from the combined LOSS instances
(loss1 + loss2).plot(subplot_columns = [['mse','fmse'],['msmse']])