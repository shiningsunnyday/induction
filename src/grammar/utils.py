"""
Wrapper around dreadnaut that computes the orbits of a graph.
NOTE: Must have installed `dreandaut`. The location of the binary can be passed
      as an argument to `compute_automorphisms`.
Author: Jean-Gabriel Young <info@jgyoung.ca>
"""
import subprocess
import networkx as nx
from os import remove

def _build_dreadnaut_file(g):
    """Prepare file to pass to dreadnaut.
    Warning
    -------
    Assumes that the nodes are represented by the 0 indexed integers.
    """
    # dreadnaut options
    file_content = ["As"]  # sparse mode
    file_content.append("-a")  # do not print out automorphisms
    file_content.append("-m")  # do not print out level markers
    # specify graph structure
    file_content.append("n=" + str(g.number_of_nodes()) + " g")
    for v in sorted(g.nodes()):
          line = " " + str(v) + " : "
          for nb in g.neighbors(v):
              if v < nb:
                  line += str(nb) + " "
          line += ";"
          file_content.append(line)
    # add nauty command
    file_content.append(".")
    file_content.append("x")
    file_content.append("o")
    return file_content


def compute_automorphisms(g, tmp_path="/home/msun415/dreadnaut.txt", dreadnaut_call="/home/msun415/nauty2_8_8/dreadnaut"):
    # get dreadnaut command file
    file_content = _build_dreadnaut_file(g)
    # write to tmp_path
    with open(tmp_path, 'w') as f:
        print("\n".join(file_content), file=f)
    # call dreadnaut
    proc = subprocess.run([dreadnaut_call],
                          input=b"< " + tmp_path.encode(),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL)
    [info, _, orbits] = proc.stdout.decode().strip().split("\n", 2)
    # ~~~~~~~~~~~~~~
    # Extract high level info from captured output
    # ~~~~~~~~~~~~~~
    num_orbits = int(info.split(" ")[0])
    num_gen = int(info.split(" ")[3])
    # ~~~~~~~~~~~~~~
    # Extract orbits
    # ~~~~~~~~~~~~~~
    # This big list comprehension splits all orbits into their own sublist, and
    # each of these orbits into individual components (as string).
    # There is still some post-processing to do since some of them are in the
    # compact notation X:X+n when the n+1 nodes of the orbits are contiguous.
    X = [_.strip().split(" (")[0].split(" ")
         for _ in orbits.replace("\n   ",'').strip().split(";")[:-1]]
    for i, orbit in enumerate(X):
        final_orbit = []
        for elem in orbit:
            if ":" in elem:
                _ = elem.split(":")
                final_orbit += range(int(_[0]), int(_[1]) + 1)
            else:
                final_orbit += [int(elem)]
        X[i] = final_orbit
    # garbage collection
    remove(tmp_path)
    return num_orbits, num_gen, X