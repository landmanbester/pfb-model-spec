def test_import():
    import pfb_model_spec

    assert hasattr(pfb_model_spec, "__version__")


def test_version_is_string():
    from pfb_model_spec import __version__

    assert isinstance(__version__, str)
