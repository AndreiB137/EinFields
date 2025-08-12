""" MIT License
Copyright (c) 2025 Andrei Bodnar (Dept of Physics and Astronomy, University of Manchester, United Kingdom), Sandeep S. Cranganore (Ellis Unit, LIT AI Lab, JKU Linz, Austria)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from general_relativity import (
    minkowski_metric,
    minkowski_metric_spherical,
    minkowski_metric_eddington_finkelstein_non_rotating,
    minkowski_metric_oblate_spheroidal,
    minkowski_metric_eddington_finkelstein_rotating,
)

from general_relativity import (
    schwarzschild_metric_spherical,
    schwarzschild_metric_spherical_distortion,
    schwarzschild_metric_kerr_schild,
    schwarzschild_metric_kerr_schild_distortion,
    schwarzschild_metric_eddington_finkelstein,
    schwarzschild_metric_eddington_finkelstein_distortion,
)

from general_relativity import (
    kerr_metric_boyer_lindquist,
    kerr_metric_boyer_lindquist_distortion,
    kerr_schild_cartesian_metric,
    kerr_schild_cartesian_metric_distortion,
    kerr_metric_eddington_finkelstein,
    kerr_metric_eddington_finkelstein_distortion,
)

from general_relativity import (
    gravitational_waves_metric,
    gravitational_waves_metric_distortion,
)

from general_relativity import (
    cartesian_to_spherical,
    cartesian_to_oblate_spheroid,
    spherical_to_kerr_schild_cartesian,
    kerr_schild_cartesian_to_spherical,
    spherical_to_cartesian,
    spherical_to_ingoing_eddington_finkelstein,
    spherical_to_outgoing_eddington_finkelstein,
    ingoing_eddington_finkelstein_to_spherical,
    eddington_finkelstein_to_boyer_lindquist,
    outgoing_eddington_finkelstein_to_spherical,
    oblate_spheroid_to_cartesian,
    oblate_spheroid_to_kerr_schild,
    kerr_schild_to_oblate_spheroid,
    kerr_schild_to_boyer_lindquist,
    boyer_lindquist_to_kerr_schild,
    boyer_lindquist_to_eddington_finkelstein,
)

metric_dict = {
    "Minkowski": {
        "coordinate_system": {
            "spherical": {
                "extra_args": [],
                "full": minkowski_metric_spherical
            },
            "cartesian": {
                "extra_args": [],
                "full": minkowski_metric
            },
            "ingoing_eddington_finkelstein_non_rotating": {
                "extra_args": [],
                "full": minkowski_metric_eddington_finkelstein_non_rotating
            },
            "oblate_spheroidal": {
                "extra_args": ["a"],
                "full": minkowski_metric_oblate_spheroidal
            },
            "ingoing_eddington_finkelstein_rotating": {
                "extra_args": ["a"],
                "full": minkowski_metric_eddington_finkelstein_rotating
            }
        }

    },
    "Schwarzschild": {
        "coordinate_system": {
            "spherical" : {
                "extra_args": ["M"],
                "full": schwarzschild_metric_spherical,
                "distortion": schwarzschild_metric_spherical_distortion
            },
            "ingoing_eddington_finkelstein": {
                "extra_args": ["M"],
                "full": schwarzschild_metric_eddington_finkelstein,
                "distortion": schwarzschild_metric_eddington_finkelstein_distortion
            },
            "kerr_schild_cartesian": {
                "extra_args": ["M"],
                "full": schwarzschild_metric_kerr_schild,
                "distortion": schwarzschild_metric_kerr_schild_distortion
            }
        }
    },
    "Kerr": {
        "coordinate_system": {
            "boyer_lindquist": {
                "extra_args": ["M", "a"],
                "full": kerr_metric_boyer_lindquist,
                "distortion": kerr_metric_boyer_lindquist_distortion
            },
            "kerr_schild_cartesian": {
                "extra_args": ["M", "a"],
                "full": kerr_schild_cartesian_metric,
                "distortion": kerr_schild_cartesian_metric_distortion
            },
            "ingoing_eddington_finkelstein": {
                "extra_args": ["M", "a"],
                "full": kerr_metric_eddington_finkelstein,
                "distortion": kerr_metric_eddington_finkelstein_distortion
            }
        }
    },
    "GW": {
        "coordinate_system": {
            "cartesian": {
                "extra_args": ["polarization_amplitudes", "omega"],
                "full": gravitational_waves_metric,
                "distortion": gravitational_waves_metric_distortion
            }
        }
    }
}

coord_transform_dict = {
    "cartesian" : {
        "spherical": {
            "extra_args": [],
            "transform": cartesian_to_spherical
        },
        "oblate_spheroid": {
            "extra_args": ["a"],
            "transform": cartesian_to_oblate_spheroid
        },
    },
    "spherical": {
        "cartesian": {
            "extra_args": [],
            "transform": spherical_to_cartesian
        },
        "ingoing_eddington_finkelstein": {
            "extra_args": ["M"],
            "transform": spherical_to_ingoing_eddington_finkelstein
        },
        "outgoing_eddington_finkelstein": {
            "extra_args": ["M"],
            "transform": spherical_to_outgoing_eddington_finkelstein
        },
        "kerr_schild_cartesian": {
            "extra_args": ["M"],
            "transform": spherical_to_kerr_schild_cartesian
        }
    },
    "ingoing_eddington_finkelstein": {
        "spherical": {
            "extra_args": ["M"],
            "transform": ingoing_eddington_finkelstein_to_spherical
        }, 
        "boyer_lindquist": {
            "extra_args": ["M", "a"],
            "transform": eddington_finkelstein_to_boyer_lindquist
        },
        "kerr_schild_cartesian": {
            "extra_args": ["M", "a"],
            "transform": lambda coords, M, a: boyer_lindquist_to_kerr_schild(
                eddington_finkelstein_to_boyer_lindquist(coords, M, a), M, a
            )
        }
    },
    "outgoing_eddington_finkelstein": {
        "spherical": {
            "extra_args": ["M"],
            "transform": outgoing_eddington_finkelstein_to_spherical
        }
    },
    "oblate_spheroid": {
        "cartesian": {
            "extra_args": ["a"],
            "transform": oblate_spheroid_to_cartesian
        },
        "kerr_schild_cartesian": {
            "extra_args": ["a"],
            "transform": oblate_spheroid_to_kerr_schild
        }
    },
    "kerr_schild_cartesian": {
        "spherical": {
            "extra_args": ["M"],
            "transform": kerr_schild_cartesian_to_spherical
        },
        "oblate_spheroid": {
            "extra_args": ["a"],
            "transform": kerr_schild_to_oblate_spheroid
        },
        "boyer_lindquist": {
            "extra_args": ["M", "a"],
            "transform": kerr_schild_to_boyer_lindquist
        },
        "ingoing_eddington_finkelstein": {
            "extra_args": ["M", "a"],
            "transform": lambda coords, M, a: kerr_schild_to_boyer_lindquist(
                boyer_lindquist_to_eddington_finkelstein(coords, M, a), M, a
            )
        }
    },
    "boyer_lindquist": {
        "kerr_schild_cartesian": {
            "extra_args": ["M", "a"],
            "transform": boyer_lindquist_to_kerr_schild
        }, 
        "ingoing_eddington_finkelstein": {
            "extra_args": ["M", "a"],
            "transform": boyer_lindquist_to_eddington_finkelstein
        }
    }
}