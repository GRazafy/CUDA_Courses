import matplotlib.pyplot as plt


y = [116.719, 10.165, 3.5, 3.4, 3.5]
x = [1024, 256, 128, 64, 32]

y2 = [461.402, 113.682, 57.019, 28.789, 14.922]


plt.plot(x, y, label="A=4096*y and B=y*4096")
plt.plot(x, y2, label="A=y*4096 and B=4096*y")
plt.xlabel('x = Time (ms)')
plt.ylabel('y = Dimension')
plt.legend()
plt.show()