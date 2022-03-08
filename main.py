import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

'''
1. indentify the area to fill and the boundary of that area
2. compute priorities
    a. P(p) = C(p)D(p)
    b. C(p) = sum of all pixels in window and omega bar (?) / len(window)
    c. D(p) = |gradient of isophote (linear structure) dot product with n_p (unit orthonormal to contour)| / a (normalization factor, usually 255)
    d. C(p) = 0 for all p in omega, 1 for all p outside of omega
3. find window with highest priority
    a. constant window size, find pixel with highest priority
4. find exemplar
    a. from source region
    b. exemplar = arg min d(window, all other regions of the source region)
    c. d(a, b) = sum squared differences of a and b
5. copy exemplar to window
6. update confidence values
    a. fill region confidence = C(p)
    b. some values will be 1 and some values will be zero
    c. confidence decays as the algorithm fills in the region, indicating that we are less confident in those values
'''

class inpainter():

    def __init__(self, window_size, img, mask):

        self.window_size = window_size
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.mask = mask
        self.img_color = img
        self.get_candidate_exemplars()
        self.bool_mask = np.ones(shape=self.img.shape, dtype=bool)
        idx = np.argwhere(mask > 0)
        self.bool_mask[idx] = True

        lap_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        self.laplacian = cv2.filter2D(mask, -1, lap_filter)
        self.dOmega = np.transpose(np.nonzero(self.laplacian > 0))


        self.conf = np.zeros(shape=self.mask.shape)
        idx = np.argwhere(self.mask<1)
        self.conf[idx] = 1

        dx_fil = np.array([[0, 0, 0],
                      [-1, 0, 1],
                      [0, 0, 0,]])

        dy_fil = np.array([[0, -1, 0],
                      [0, 0, 0],
                      [0, 1, 0]])

        self.dx = np.zeros(shape=self.img.shape)
        self.dy = np.zeros(shape=self.img.shape)
        for channel in range(self.img_color.shape[2]):
            gauss = cv2.blur(self.img_color[:, :, channel], (5,5))
            self.dx += cv2.filter2D(gauss, -1, dx_fil) / 3
            self.dy += cv2.filter2D(gauss, -1, dy_fil) / 3

        self.max_grad_idx = self.get_max_grad(self.dx, self.dy)

        ## The normal term is weird. I believe I can get a decent approximation by searching the area surrounding
        ## a pixel p for the 2 darkest points other than p and assuming them to be the points before and after p.
        ## This should give us enough to compute the slope between them and an orthoganal vector between them
        ## can be calculated.


        self.dOmega_to_normal = {}
        for p in self.dOmega:
            i, j = p
            self.dOmega_to_normal[(i, j)] = self.get_normal(p, self.laplacian)


        self.data = np.zeros(shape=self.img.shape)
        for key in self.dOmega_to_normal.keys():
            i, j = key
            normal = self.dOmega_to_normal[key]
            dx, dy = self.dx[i, j], self.dy[i, j]
            grad = np.array([dx, dy])
            self.data[i, j] = np.abs(np.dot(grad, normal)) / 255

        self.prio = self.conf * self.data

    def get_highest_priority(self):
        best_coords = self.dOmega[0]
        best_prio = -1
        for coords in self.dOmega:
            if self.prio[coords[0], coords[1]] > best_prio:
                best_coords = coords
        return best_coords

    def get_normal(self, p, img):
        i, j = p
        patch = np.copy(img[i - 1:i + 2, j - 1:j + 2])
        patch[1, 1] = 0
        max1 = np.unravel_index(np.argmax(patch), shape=(3, 3))
        patch[max1] = 0
        max2 = np.unravel_index(np.argmax(patch), shape=(3, 3))
        dy = max1[0] - max2[0]
        dx = max1[1] - max2[1]
        norm = ((dy ** 2) + (dx ** 2)) ** .5
        if norm == 0:
            norm = 1
        return np.array([dy / norm, -1 * dx / norm])

    def get_candidate_exemplars(self):
        exemplar_candidates = []
        for i in range(self.window_size // 2, self.img.shape[0] - self.window_size//2, 1):
            for j in range(self.window_size // 2, self.img.shape[1] - self.window_size//2, 1):
                patch_sum = np.sum(self.get_patch((i,j), self.mask))
                if patch_sum == 0:
                    exemplar_candidates += [self.get_patch((i, j), self.img_color)]
        self.exemplar_candidates = exemplar_candidates

    def get_exemplar(self, p):
        ## p is a point to get the exemplar, not a patch

        def SSD(p, q):
            p_patch = self.get_patch(p, self.img_color)
            p_mask = self.get_patch(p, self.bool_mask)
            idx = np.argwhere(p_mask is False)
            p_patch[idx] = 0
            # q[idx] = 0

            reg = 1 - (idx.shape[0] / (self.window_size ** 2))
            # reg = 1



            p_blue, q_blue = p_patch[:, :, 0], q[:, :, 0]
            p_green, q_green = p_patch[:, :, 1], q[:, :, 1]
            p_red, q_red = p_patch[:, :, 2], q[:, :, 2]

            red_diff = np.square(q_red * reg - p_red)
            blue_diff = np.square(q_blue * reg - p_blue)
            green_diff = np.square(q_green * reg - p_green)

            return np.sum(red_diff) + np.sum(blue_diff) + np.sum(green_diff)

        best_min = float('inf')
        best_patch = np.zeros(shape=(self.window_size, self.window_size))
        for candidate_patch in self.exemplar_candidates:
            q = np.copy(candidate_patch)
            cur_ssd = SSD(p, q)
            if cur_ssd < best_min:
                best_min = cur_ssd
                best_patch = candidate_patch
        print('best SSD: ', best_min)
        return best_patch

    def update_dOmega(self, p):
        i, j = p
        lower_i = i - self.window_size // 2
        upper_i = i + self.window_size // 2
        lower_j = j - self.window_size // 2
        upper_j = j + self.window_size // 2
        self.mask[lower_i:upper_i, lower_j:upper_j] = 0

        lap_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        self.laplacian = cv2.filter2D(self.mask, -1, lap_filter)
        dOmega = np.transpose(np.nonzero(self.laplacian > 0))
        self.dOmega = dOmega

    def get_max_grad(self, gradient_x, gradient_y):
        grad_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
        out = np.zeros(shape=(gradient_x.shape[0], gradient_x.shape[1], 2))
        for i in range(self.window_size//2, gradient_x.shape[0]-self.window_size//2, 1):
            for j in range(self.window_size//2, gradient_x.shape[1]-self.window_size//2, 1):
                lower_i = i - self.window_size // 2
                upper_i = i + self.window_size // 2
                lower_j = j - self.window_size // 2
                upper_j = j + self.window_size // 2
                flattened_idx = np.argmax(grad_mag[lower_i:upper_i, lower_j:upper_j]).astype(int)
                unraveled = np.asarray(np.unravel_index(flattened_idx, shape=(self.window_size, self.window_size)))
                unraveled += np.array([i, j])
                out[i, j, :] = unraveled
        return out

    def update_priorities(self, p, out):
        ## P is a point, not a patch

        i, j = p
        lower_i = i - self.window_size // 2
        upper_i = i + self.window_size // 2
        lower_j = j - self.window_size // 2
        upper_j = j + self.window_size // 2
        patch = self.conf[lower_i:upper_i, lower_j:upper_j]
        self.conf[i, j] = np.sum(patch) / (self.window_size ** 2)


        dx_fil = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1,]]) / 3

        dy_fil = np.array([[-1, -1,-1],
                      [0, 0, 0],
                      [1, 1, 1]]) / 3

        self.dx = np.zeros(shape=self.img.shape)
        self.dy = np.zeros(shape=self.img.shape)
        for channel in range(out.shape[2]):
            gauss = cv2.blur(out[:, :, channel], (5,5))
            self.dx += cv2.filter2D(gauss, -1, dx_fil) / 3
            self.dy += cv2.filter2D(gauss, -1, dy_fil) / 3

        self.dOmega_to_normal = {}
        for p in self.dOmega:
            i, j = p
            self.dOmega_to_normal[(i, j)] = self.get_normal(p, self.laplacian)

        self.data = np.zeros(shape=self.img.shape)
        for key in self.dOmega_to_normal.keys():
            i, j = key
            normal = self.dOmega_to_normal[key]
            max_grad_i, max_grad_j = self.max_grad_idx[i, j].astype(int)
            dx, dy = self.dx[max_grad_i, max_grad_j], self.dy[max_grad_i, max_grad_j]
            grad = np.array([dy, dx])
            self.data[i, j] = np.abs(np.dot(grad, normal)) / 255
        self.prio = self.conf * self.data

    def get_patch(self, p, img):
        i, j = p
        lower_i = i - self.window_size // 2
        upper_i = i + self.window_size // 2
        lower_j = j - self.window_size // 2
        upper_j = j + self.window_size // 2
        try:
            return np.copy(img[lower_i:upper_i, lower_j:upper_j, :])
        except IndexError:
            return np.copy(img[lower_i:upper_i, lower_j:upper_j])

    def inpaint(self):
        out = np.copy(self.img_color)
        count = 0
        while len(self.dOmega) != 0:
            # if count % 100 == 0:
            #     fig = plt.figure(figsize=(10, 7))
            #     fig.add_subplot(1, 2, 1)
            #     plt.imshow(out, cmap='gray')
            #     fig.add_subplot(1,2,2)
            #     plt.imshow(self.mask, cmap='gray')
            #     plt.show()

            count += 1

            p = self.get_highest_priority()

            grad = self.max_grad_idx[tuple(p)].astype(int)
            dx = self.dx[tuple(grad)]
            dy = self.dy[tuple(grad)]
            normal = self.dOmega_to_normal[tuple(p)]
            data = self.data[tuple(p)]
            conf = self.conf[tuple(p)]
            prio = self.prio[tuple(p)]


            print('grad: ', (dx, dy), 'p: ', p, 'normal: ', normal, 'fill front length: ', len(self.dOmega))
            print('data: ', data, 'conf: ', conf, 'prio: ', prio)
            print('dot product: ', np.abs(np.dot(np.array([dy, dx]), normal)) / 255)
            q = self.get_exemplar(p)
            i, j = p
            lower_i = i - self.window_size // 2
            upper_i = i + self.window_size // 2
            lower_j = j - self.window_size // 2
            upper_j = j + self.window_size // 2

            out[lower_i:upper_i, lower_j:upper_j] = q
            self.update_dOmega(p)
            self.update_priorities(p, out)

        return out


def main():
    ## open image
    ## figure out how to load mask?
    img = plt.imread('test/image1.jpg')
    mask = plt.imread('test/mask1.jpg')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    inp = inpainter(10, img, mask)
    res = inp.inpaint()
    im = np.hstack((img, res))
    plt.imshow(im, cmap='gray')
    plt.show()

    img = plt.imread('test/image4.jpg')
    mask = plt.imread('test/mask4.jpg')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    inp = inpainter(20, img, mask)
    res = inp.inpaint()
    im = np.hstack((img, res))
    plt.imshow(im, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()


