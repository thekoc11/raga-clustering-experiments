import numpy as np

class Viewpoints:
    def __init__(self, chroma):
        self.funcs = {'pitch': self.get_pitches,
                      'interval': self._calculate_interval,
                      'duration': self.get_dur,
                      'onsets': self.get_ons,
                      'ioi': self._calculate_ioi,
                      # 'reference_interval': self._calculate_intref,
                      'pitch_contour': self._calculate_pcontour,
                      'duration_contour': self._calculate_dcontour,
                      'duration_ratio': self._calculate_dratio,
                      'pcd': self._compute_pcd,
                      'weighted_pcd': self._compute_pcd_weighted,
                      }
        self.chroma = chroma
        self.dur = []
        self.ons = []
        self.pitches = self._calculate_pitches()

    def _calculate_pitches(self):
        pitches = self.chroma.T.argmax(axis=1)
        pitches_refined = []
        dur = []
        ons = []
        pitches_refined.append(pitches[0])
        dur.append(1)
        ons.append(0)
        for i in range(1, len(pitches)):
            if pitches[i - 1] != pitches[i]:
                pitches_refined.append(pitches[i])
                dur.append(1)
                ons.append(i)
            else:
                dur[-1] += 1
        self.dur = np.array(dur)
        self.ons = np.array(ons)
        return np.array(pitches_refined, dtype='int32')

    def get_pitches(self):
        # print(f"max value in pitches: {self.pitches.max()}")
        return self.pitches

    def get_dur(self):
        return self.dur
    def  get_ons(self):
        return self.ons

    def _calculate_interval(self):
        int = np.zeros(len(self.pitches))
        for i in range(1, len(self.pitches)):
            int[i] = self.pitches[i] - self.pitches[i-1]
            if self.pitches[i] == 12:
                int[i] = 0
        return int.astype('int32')

    def _calculate_ioi(self):
        ioi = np.zeros(len(self.ons))
        for i in range(1, len(self.ons)):
            ioi[i] = self.ons[i] - self.ons[i-1]
        return ioi.astype('int32')

    def _calculate_pcontour(self):
        pcontour = np.zeros(len(self.pitches)).astype('bool')
        if len(self.pitches) < 2:
            return [None]
        pcontour[0] = self.pitches[0] <= self.pitches[1]
        for i in range(1, len(self.pitches)):
            if self.pitches[i] == 12:
                pcontour[i] = pcontour[i-1]
            pcontour[i] = self.pitches[i - 1] < self.pitches[i]
        return pcontour

    def _calculate_dcontour(self):
        dcontour = np.zeros(len(self.dur)).astype('int32')
        # dcontour[0] = self.dur[0] < self.pitches[1]
        if len(self.dur) < 2:
            return [-9999]
        for i in range(1, len(self.dur)):
            if self.dur[i - 1] != self.dur[i]:
                if self.dur[i-1] > self.dur[i]:
                    dcontour[i] = 0
                else:
                    dcontour[i] = 1
            else:
                dcontour[i] = dcontour[i-1]
        dcontour[0] = dcontour[1]
        return dcontour

    def _compute_pcd(self):
        tot_events = self.pitches.shape[0]
        pcd = np.zeros(13, dtype='float64')
        for p in self.pitches:
            pcd[p] = pcd[p] + 1
        pcd /= tot_events
        return pcd

    def _compute_pcd_weighted(self):
        tot_dur = self.dur.sum()
        pcd = np.zeros(13, dtype='float64')
        for i in range(len(self.pitches)):
            pcd[self.pitches[i]] = pcd[self.pitches[i]] + self.dur[i]
        pcd /= tot_dur

        return pcd

    def _calculate_dratio(self):
        dratios = np.zeros(len(self.dur))
        for i in range(1, len(self.dur)):
            if self.pitches[i-1] == 12:
                dratios[i] = float(self.dur[i] / self.dur[i-2])
            else:
                dratios[i] = float(self.dur[i] / self.dur[i-1])
        return dratios

    def get_viewpoint(self, viewpoint):
        # if viewpoint != 'all':
        retval = self.funcs[viewpoint]()
        # else:
        #     vps = {}

        return retval


    def link_viewpoints(self, viewpoint1, viewpoint2):
        retval = zip(self.funcs[viewpoint1](), self.funcs[viewpoint2]())
        return retval

    def backreference_pitches(self):
        vp = self.pitches
        br = np.zeros_like(vp)
        for i in range(1, len(vp)):
            for j in range(i):
                if vp[j] == vp[i]:
                    br[i] = -(self.dur[:i].sum() - self.dur[:j].sum())
        return br

    def scale_sensitive_params(self):
        events = (self.chroma, self.pitches)
        dists = (self._compute_pcd(), self._compute_pcd_weighted())

        return events, dists


def main():
    X = np.random.uniform(0, 3, size=[13, 20])
    print(X.T.argmax(axis=1))
    vp = Viewpoints(X)
    events, dists = vp.scale_sensitive_params()
    print(events[0].shape, events[1].shape)
    print(dists[0].shape, dists[1].shape)
    # print(vp.pitches, vp.pitches.shape)
    # print(vp.get_viewpoint('augmented_pitches'), vp.get_viewpoint('augmented_pitches').shape)
    # print(vp.get_viewpoint('pitch_contour'), vp.get_viewpoint('pitch_contour').shape)
    # print(vp.backreference_pitches(), vp.backreference_pitches().shape)
    # print(dict(vp.link_viewpoints('interval', 'ioi')))




if __name__=='__main__':
    main()