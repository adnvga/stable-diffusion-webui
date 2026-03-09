import modules.script_callbacks as script_callbacks

def on_app_started(demo, app):
    print("🔥 [ws_safe] Backend extension loaded successfully.")
    print("🚨🚨🚨 WS_SAFE VERSION 6001 🚨🚨🚨");


script_callbacks.on_app_started(on_app_started)
