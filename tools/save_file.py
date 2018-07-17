import os


class SaverInfo:
    save_type = None
    extra_info = ''
    _save_num = None

    def __init__(self, save_type: str, save_num=None, extra_info=''):
        self.save_type = save_type
        self.extra_info = extra_info

        if save_num is not None:
            assert isinstance(save_num, int) and 0 <= save_num <= 9999, "`save_num` must be from 0 ~ 9999"
            self._save_num = save_num
        else:
            run_nums = [int(name.split('_')[0]) for name in os.listdir("../{}".format(save_type))]
            self._save_num = 0 if len(run_nums) == 0 else max(run_nums) + 1
            assert 0 <= self._save_num <= 9999, "Hmm, this is weird, but it seems like the last file saved to " \
                                                "the {} folder began with 9999 (quit doing so much work!)".format(save_type)

    @property
    def save_num(self):
        return self._save_num

    def get_file_name(self):
        save_num_as_str = "{:0>4s}".format(str(self._save_num))
        extra_info = self.extra_info
        if extra_info:
            extra_info = '_{}'.format(extra_info)
        return "../{}/{}{}".format(self.save_type, save_num_as_str, extra_info)
