class PValueProcessor:
    def __init__(self, spearmans_dic):
        self.spearmans_dic = spearmans_dic

    def _calculate_ratio(self, key, part):
        values = [value[0] for value in self.spearmans_dic[key][part] if len(value) > 1 and value[1] < 0.05]
        total = len(values) + len([value[0] for value in self.spearmans_dic[key][part] if len(value) > 1 and value[1] > 0.05])
        return len(values) / total if total else 0

    def calculate_RNFL_x_l(self):
        return self._calculate_ratio('RNFL', 'left')

    def calculate_PR_RPE_x_l(self):
        return self._calculate_ratio('Photoreceptors + RPE', 'left')
