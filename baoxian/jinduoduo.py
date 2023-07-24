import math

if __name__ == '__main__':
    values = [
        101970,
        105029,
        108180,
        111425,
        114768,
        119241,
        122818,
        126503,
        130298,
        134207,
        138233,
        142380,
        146652,
        151051,
        155583,
        160250,
        165058,
        170009,
        175110,
        180363,
        185774,
        191347,
        197088,
        203000,
        209090,
        215363,
        221824,
        228478,
        235333,
        242393,
    ]
    in_base = 10
    start_age = 30

    for idx, value in enumerate(values, start=1):
        value = value / 10000
        fu_ratio = math.pow(value / in_base, 1 / idx) - 1  #复利
        dan_ratio = (value / in_base - 1) / idx
        print(f"{start_age + idx}岁: 复利: {fu_ratio:5f}, 单利: {dan_ratio:5f}")