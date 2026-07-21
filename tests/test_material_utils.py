import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from material_utils import force_matte_node_tree


# ── Minimal fakes mimicking the bpy node-tree API surface we touch ───────────

class FakeLink:
    def __init__(self, name):
        self.name = name


class FakeSocket:
    def __init__(self, default_value=0.0, links=()):
        self.default_value = default_value
        self.links = list(links)


class FakeNode:
    def __init__(self, type_, inputs):
        self.type = type_
        self.inputs = inputs


class FakeTree:
    def __init__(self, nodes):
        self.nodes = nodes
        self.removed = []
        self.links = self

    def remove(self, link):
        self.removed.append(link)
        for node in self.nodes:
            for sock in node.inputs.values():
                if link in sock.links:
                    sock.links.remove(link)


def _principled(roughness=0.1, metallic=1.0, spec_name="Specular IOR Level",
                roughness_links=()):
    return FakeNode('BSDF_PRINCIPLED', {
        "Roughness": FakeSocket(roughness, roughness_links),
        "Metallic": FakeSocket(metallic),
        spec_name: FakeSocket(0.5),
        "Base Color": FakeSocket((1, 1, 1, 1)),
    })


# ── Tests ────────────────────────────────────────────────────────────────────

def test_sets_roughness_metallic_specular():
    node = _principled(roughness=0.05, metallic=1.0)
    tree = FakeTree([node])
    changed, cut = force_matte_node_tree(tree, 0.9)
    assert changed == 1 and cut == 0
    assert node.inputs["Roughness"].default_value == 0.9
    assert node.inputs["Metallic"].default_value == 0.0
    assert node.inputs["Specular IOR Level"].default_value == 0.0


def test_cuts_texture_driven_roughness_link():
    """The case a plain roughness-floor silently skips."""
    link = FakeLink('tex->roughness')
    node = _principled(roughness=0.02, roughness_links=[link])
    tree = FakeTree([node])
    changed, cut = force_matte_node_tree(tree, 0.9)
    assert cut == 1
    assert link in tree.removed
    assert node.inputs["Roughness"].links == []
    assert node.inputs["Roughness"].default_value == 0.9


def test_handles_legacy_specular_socket_name():
    node = _principled(spec_name="Specular")
    tree = FakeTree([node])
    force_matte_node_tree(tree, 0.8)
    assert node.inputs["Specular"].default_value == 0.0


def test_ignores_non_principled_nodes():
    other = FakeNode('TEX_IMAGE', {"Vector": FakeSocket(0.0)})
    tree = FakeTree([other])
    changed, cut = force_matte_node_tree(tree, 0.9)
    assert changed == 0 and cut == 0
    assert other.inputs["Vector"].default_value == 0.0


def test_multiple_principled_nodes_all_changed():
    nodes = [_principled(), _principled()]
    tree = FakeTree(nodes)
    changed, _ = force_matte_node_tree(tree, 0.7)
    assert changed == 2
    for n in nodes:
        assert n.inputs["Roughness"].default_value == 0.7


def test_missing_socket_is_tolerated():
    node = FakeNode('BSDF_PRINCIPLED', {"Roughness": FakeSocket(0.1)})
    tree = FakeTree([node])
    changed, _ = force_matte_node_tree(tree, 0.9)
    assert changed == 1
    assert node.inputs["Roughness"].default_value == 0.9


def test_empty_tree():
    changed, cut = force_matte_node_tree(FakeTree([]), 0.9)
    assert changed == 0 and cut == 0
