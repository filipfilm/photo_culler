"""ON1 Photo RAW integration.

ON1 has no plugin API for this kind of work -- it hosts Photoshop-format pixel plugins
and ships a Lightroom .lrplugin, neither of which can be handed a selection of originals
and asked to write metadata. What it does have is "Send to Other Application", so the
integration is a small macOS application bundle that receives the selected files and
opens the review popup. See install.py for how the bundle is built.
"""
