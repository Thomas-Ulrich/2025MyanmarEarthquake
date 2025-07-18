#!/usr/bin/env python3

import easi
import seissolxdmf
import seissolxdmfwriter as sxw
import argparse
import numpy as np


class SeissolxdmfExtended(seissolxdmf.seissolxdmf):
    def __init__(self, xdmfFilename):
        super().__init__(xdmfFilename)
        self.xyz = self.ReadGeometry()
        self.connect = self.ReadConnect()

    def ReadFaultTags(self):
        """Read partition array"""
        return self.Read1dData("fault-tag", self.nElements, isInt=True).T

    def ComputeCellCenters(self):
        """compute cell center array"""
        return (
            self.xyz[self.connect[:, 0]]
            + self.xyz[self.connect[:, 1]]
            + self.xyz[self.connect[:, 2]]
        ) / 3.0

    def ComputeCellNormals(self, ref_vector):
        un = np.cross(
            self.xyz[self.connect[:, 1], :] - self.xyz[self.connect[:, 0], :],
            self.xyz[self.connect[:, 2], :] - self.xyz[self.connect[:, 0], :],
        )
        norm = np.apply_along_axis(np.linalg.norm, 1, un)
        un = un / norm[:, np.newaxis]
        mysign = np.sign(np.dot(un, ref_vector))
        un = un * mysign[:, np.newaxis]
        return un

    def ComputeCellArea(self):
        cross0 = np.cross(
            self.xyz[self.connect[:, 1], :] - self.xyz[self.connect[:, 0], :],
            self.xyz[self.connect[:, 2], :] - self.xyz[self.connect[:, 0], :],
        )
        return 0.5 * np.apply_along_axis(np.linalg.norm, 1, cross0)


def compute_tractions(dicStress, un):
    # compute Strike and dip direction
    us = np.zeros(un.shape)
    us[:, 0] = -un[:, 1]
    us[:, 1] = un[:, 0]
    norm = np.apply_along_axis(np.linalg.norm, 1, us)
    us = us / norm[:, np.newaxis]
    ud = np.cross(un, us)

    def compute_traction_vector(dicStress, un):
        nel = un.shape[0]
        Tractions = np.zeros((nel, 3))
        a = np.stack((dicStress["s_xx"], dicStress["s_xy"], dicStress["s_xz"]), axis=1)
        Tractions[:, 0] = np.sum(a * un, axis=1)
        a = np.stack((dicStress["s_xy"], dicStress["s_yy"], dicStress["s_yz"]), axis=1)
        Tractions[:, 1] = np.sum(a * un, axis=1)
        a = np.stack((dicStress["s_xz"], dicStress["s_yz"], dicStress["s_zz"]), axis=1)
        Tractions[:, 2] = np.sum(a * un, axis=1)
        return Tractions

    Tractions = compute_traction_vector(dicStress, un)

    outDict = {}
    # compute Traction components
    outDict["T_n"] = np.sum(Tractions * un, axis=1)
    outDict["T_s"] = np.sum(Tractions * us, axis=1)
    outDict["T_d"] = np.sum(Tractions * ud, axis=1)
    return outDict


if __name__ == "__main__":
    # parsing python arguments
    parser = argparse.ArgumentParser(
        description=(
            " compute fracture energy from easi/yaml file and fault output file"
        )
    )
    parser.add_argument("fault_filename", help="fault.xdmf filename")
    parser.add_argument("yaml_filename", help="fault easi/yaml filename")

    parser.add_argument(
        "--ref_vector",
        nargs=1,
        help="reference vector (see seissol parameter file) used to choose fault normal (coma separated string)",
    )
    parser.add_argument(
        "--output_file",
        help="path and prefix of the output file",
        nargs=1,
        default=["initial-stress-fault"],
    )
    args = parser.parse_args()

    sx = SeissolxdmfExtended(args.fault_filename)
    centers = sx.ComputeCellCenters()
    tags = sx.ReadFaultTags()

    parameters = "mu_s,d_c,T_n".split(",")
    try:
        out = easi.evaluate_model(centers, tags, parameters, args.yaml_filename)
    except ValueError:
        # T_n not given: computing from s_ij then
        if not args.ref_vector:
            raise ValueError(
                "ref_vector has to be defined for computing tractions from stress"
            )
        else:
            ref_vector = [float(v) for v in args.ref_vector[0].split(",")]
        print(f"[T_s, T_d, T_n] not found in {args.yaml_filename}, using s_ij")
        dicStress = easi.evaluate_model(
            centers,
            tags,
            ["s_xx", "s_yy", "s_zz", "s_xy", "s_xz", "s_yz"],
            args.yaml_filename,
        )
        normals = sx.ComputeCellNormals(ref_vector)
        out = compute_tractions(dicStress, normals)
        parameters = "mu_s,d_c".split(",")
        out |= easi.evaluate_model(centers, tags, parameters, args.yaml_filename)

    available = sx.ReadAvailableDataFields()
    assert "ASl" in available
    assert "Mud" in available
    out |= {
        "ASl": sx.ReadData("ASl", sx.ndt - 1),
        "Mud": sx.ReadData("Mud", sx.ndt - 1),
    }

    face_area = sx.ComputeCellArea()
    mus = out["mu_s"]
    mud = out["Mud"]
    # remove negative slip (Gibbs effects)
    slip = np.maximum(0, out["ASl"])
    Pn0 = out["T_n"]
    dc = np.minimum(out["d_c"], slip)

    out["G_c"] = -(mus - mud) * dc * Pn0 * 0.5

    slip_thres = 0.01
    id_slip = np.where(slip > slip_thres)
    ruptured_area = np.sum(face_area[id_slip])

    G_c = np.sum(out["G_c"][id_slip] * face_area[id_slip] / ruptured_area)
    print(f"fracture energy from based on delta_mu * dc * sigma_n: {G_c:e}")

    average_slip = np.sum(face_area[id_slip] * slip[id_slip])
    average_slip /= ruptured_area
    print(f"average slip {average_slip}")
    # keep only G_c
    out = {key: value for key, value in out.items() if key == "G_c"}

    sxw.write(
        args.output_file[0],
        sx.xyz,
        sx.connect,
        out,
        {"0.0": 0},
        reduce_precision=True,
        backend="raw",
    )
