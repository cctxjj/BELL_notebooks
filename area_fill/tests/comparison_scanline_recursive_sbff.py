from area_fill.fill_algs import scanline_stackbased_flood_fill_4con, recursive_stackbased_flood_fill_4con
import util.image_handler as imgutil
import time

sum_1 = 0
sum_2 = 0
for i in range(10000):
    print(i)
    ar = imgutil.format_greyscale_img("imgs//test_img_5.png")
    start_1 = time.perf_counter()
    scanline_stackbased_flood_fill_4con(ar, (255, 181), 140)
    end_1 = time.perf_counter()
    sum_1 += end_1 - start_1

    ar = imgutil.format_greyscale_img("imgs//test_img_5.png")
    start_2 = time.perf_counter()
    recursive_stackbased_flood_fill_4con(ar, (255, 181), 140)
    end_2 = time.perf_counter()
    sum_2 += end_2 - start_2

print(sum_1/10000)
print(sum_2/10000)

# Todo: improve, save data automatically to csv