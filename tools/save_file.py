import os


class SaveInfo:
    save_type = None
    extra_info = ''
    _save_num = None

    def __init__(self, save_type: str, save_num=None, extra_info=''):
        self.save_type = save_type
        self.extra_info = extra_info

        if type(save_num) is int:
            assert 0 <= save_num <= 9999, "`save_num` must be from 0 ~ 9999"
            self._save_num = save_num

            # Gets the matching file name based off of `save_num`
            if extra_info == '':
                save_num_as_str = self.save_num_to_str()
                files_with_num = [i for i in os.listdir(self._get_dir()) if i.startswith(save_num_as_str)]
                assert len(files_with_num) == 1, "Exactly one file should begin with '{}' if trying to load a save file".format(save_num_as_str)
                file_name = files_with_num[0]
                file_pieces = file_name.split('_', 1)
                self._save_name = self._save_name = "{}/{}".format(self._get_dir(), file_name)
                if len(file_pieces) == 2:
                    self.extra_info = file_pieces[1]
        elif save_num is None:
            run_nums = [int(name.split('_')[0]) for name in os.listdir(self._get_dir()) if 'checkpoint' not in name]
            self._save_num = 0 if len(run_nums) == 0 else max(run_nums) + 1
            assert 0 <= self._save_num <= 9999, "Hmm, this is weird, but it seems like the last file saved to " \
                                                "the {} folder was at least 9999 (quit doing so much work!)".format(save_type)

    def save_num_to_str(self):
        return "{:0>4s}".format(str(self._save_num))

    @property
    def save_num(self):
        return self._save_num

    def _get_dir(self):
        return os.path.join(os.path.dirname(__file__), '../saves/{}'.format(self.save_type))

    def get_file_name(self):
        if not hasattr(self, '_save_name'):
            save_num_as_str = ''
            if self._save_num is not None:
                save_num_as_str = self.save_num_to_str()
            extra_info = self.extra_info
            if extra_info and self._save_num is not None:
                extra_info = '_{}'.format(extra_info)
            self._save_name = "{}/{}{}".format(self._get_dir(), save_num_as_str, extra_info)

        return self._save_name
