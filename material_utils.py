"""Material manipulation helpers, decoupled from bpy for testability.

The functions here take duck-typed Blender node trees (anything exposing
`.nodes`, `.links`, and node `.inputs` with `.links` / `.default_value`), so the
logic can be unit-tested without a Blender runtime — same rationale as
latent_utils.py being split out of generate_traversals.py.
"""

# Blender renamed the Principled BSDF specular socket in 4.x; try both.
_SPECULAR_SOCKETS = ("Specular IOR Level", "Specular")

# Sockets forced to 0 to remove mirror-like response. Roughness is handled
# separately since its target value is caller-supplied.
_ZERO_SOCKETS = ("Metallic",)


def force_matte_node_tree(tree, roughness):
    """Force every Principled BSDF in `tree` to a matte finish.

    Sets Roughness to `roughness`, Metallic to 0, and Specular to 0 — first
    REMOVING any link driving those sockets. Cutting the links is the point:
    a texture-driven Roughness cannot be overridden by writing default_value
    alone, because the link keeps winning, which is why a plain roughness-floor
    silently skips exactly the glossy materials it should be fixing.

    Returns (n_nodes_changed, n_links_cut). A high n_links_cut means many
    materials were texture-driven and were being missed before.
    """
    n_changed = 0
    n_links_cut = 0
    for node in getattr(tree, 'nodes', []):
        if getattr(node, 'type', None) != 'BSDF_PRINCIPLED':
            continue

        targets = [("Roughness", roughness)]
        targets += [(name, 0.0) for name in _ZERO_SOCKETS]
        for spec in _SPECULAR_SOCKETS:
            if spec in node.inputs:
                targets.append((spec, 0.0))
                break

        touched = False
        for name, value in targets:
            inp = node.inputs.get(name)
            if inp is None:
                continue
            for link in list(getattr(inp, 'links', [])):
                tree.links.remove(link)
                n_links_cut += 1
            try:
                inp.default_value = value
            except (TypeError, ValueError):
                continue
            touched = True
        if touched:
            n_changed += 1
    return n_changed, n_links_cut
