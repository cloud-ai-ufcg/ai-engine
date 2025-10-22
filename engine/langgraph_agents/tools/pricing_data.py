aws_instance_types = [
    # T4g family (ARM-based burstable)
    {"name": "t4g.nano", "vcpus": 2, "memory_gb": 0.5, "price_usd_per_hour": 0.0042, "provider": "AWS"},
    {"name": "t4g.micro", "vcpus": 2, "memory_gb": 1, "price_usd_per_hour": 0.0084, "provider": "AWS"},
    {"name": "t4g.small", "vcpus": 2, "memory_gb": 2, "price_usd_per_hour": 0.0168, "provider": "AWS"},
    {"name": "t4g.medium", "vcpus": 2, "memory_gb": 4, "price_usd_per_hour": 0.0336, "provider": "AWS"},
    {"name": "t4g.large", "vcpus": 2, "memory_gb": 8, "price_usd_per_hour": 0.0672, "provider": "AWS"},
    {"name": "t4g.xlarge", "vcpus": 4, "memory_gb": 16, "price_usd_per_hour": 0.1344, "provider": "AWS"},
    {"name": "t4g.2xlarge", "vcpus": 8, "memory_gb": 32, "price_usd_per_hour": 0.2688, "provider": "AWS"},

    # T3 family (Intel burstable)
    {"name": "t3.nano", "vcpus": 2, "memory_gb": 0.5, "price_usd_per_hour": 0.0052, "provider": "AWS"},
    {"name": "t3.micro", "vcpus": 2, "memory_gb": 1, "price_usd_per_hour": 0.0104, "provider": "AWS"},
    {"name": "t3.small", "vcpus": 2, "memory_gb": 2, "price_usd_per_hour": 0.0208, "provider": "AWS"},
    {"name": "t3.medium", "vcpus": 2, "memory_gb": 4, "price_usd_per_hour": 0.0416, "provider": "AWS"},
    {"name": "t3.large", "vcpus": 2, "memory_gb": 8, "price_usd_per_hour": 0.0832, "provider": "AWS"},
    {"name": "t3.xlarge", "vcpus": 4, "memory_gb": 16, "price_usd_per_hour": 0.1664, "provider": "AWS"},
    {"name": "t3.2xlarge", "vcpus": 8, "memory_gb": 32, "price_usd_per_hour": 0.3328, "provider": "AWS"},

    # T3a family (AMD burstable)
    {"name": "t3a.nano", "vcpus": 2, "memory_gb": 0.5, "price_usd_per_hour": 0.0047, "provider": "AWS"},
    {"name": "t3a.micro", "vcpus": 2, "memory_gb": 1, "price_usd_per_hour": 0.0094, "provider": "AWS"},
    {"name": "t3a.small", "vcpus": 2, "memory_gb": 2, "price_usd_per_hour": 0.0188, "provider": "AWS"},
    {"name": "t3a.medium", "vcpus": 2, "memory_gb": 4, "price_usd_per_hour": 0.0376, "provider": "AWS"},
    {"name": "t3a.large", "vcpus": 2, "memory_gb": 8, "price_usd_per_hour": 0.0752, "provider": "AWS"},
    {"name": "t3a.xlarge", "vcpus": 4, "memory_gb": 16, "price_usd_per_hour": 0.1504, "provider": "AWS"},
    {"name": "t3a.2xlarge", "vcpus": 8, "memory_gb": 32, "price_usd_per_hour": 0.3008, "provider": "AWS"},

    # T2 family (Intel burstable - previous generation)
    {"name": "t2.nano", "vcpus": 1, "memory_gb": 0.5, "price_usd_per_hour": 0.0058, "provider": "AWS"},
    {"name": "t2.micro", "vcpus": 1, "memory_gb": 1, "price_usd_per_hour": 0.0116, "provider": "AWS"},
    {"name": "t2.small", "vcpus": 1, "memory_gb": 2, "price_usd_per_hour": 0.023, "provider": "AWS"},
    {"name": "t2.medium", "vcpus": 2, "memory_gb": 4, "price_usd_per_hour": 0.0464, "provider": "AWS"},
    {"name": "t2.large", "vcpus": 2, "memory_gb": 8, "price_usd_per_hour": 0.0928, "provider": "AWS"},
    {"name": "t2.xlarge", "vcpus": 4, "memory_gb": 16, "price_usd_per_hour": 0.1856, "provider": "AWS"},
    {"name": "t2.2xlarge", "vcpus": 8, "memory_gb": 32, "price_usd_per_hour": 0.3712, "provider": "AWS"},
]