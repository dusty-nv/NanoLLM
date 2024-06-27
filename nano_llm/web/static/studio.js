
// https://www.geeksforgeeks.org/how-to-include-a-javascript-file-in-another-javascript-file/
function include(file) {
    let script = document.createElement('script');
    
    script.src = file;
    script.type = 'text/javascript';
    script.defer = true;
 
    document.getElementsByTagName('head').item(0).appendChild(script);
}

include('https://cdnjs.cloudflare.com/ajax/libs/gridstack.js/10.1.2/gridstack-all.min.js');
include('https://cdn.jsdelivr.net/gh/jerosoler/Drawflow/dist/drawflow.min.js');

include('/static/jquery-3.6.3.min.js');
include('/static/bootstrap.bundle.min.js');
include('/static/select2.min.js');
        
include('/static/webrtc.js');
include('/static/websocket.js');
include('/static/debounce.js');
include('/static/audio.js');
include('/static/grid.js');
include('/static/menu.js');
include('/static/alert.js');


function loadStudio(config) {
  connectWebsocket(onWebsocketMsg, port=config.ws_port);
        
  enumerateAudioDevices();
  openAudioDevices();

  addGrid();
  
  //setTimeout( ()=>{ debugger }, 5000);
}

function onWebsocketMsg(payload, type) {
  if( type == MESSAGE_JSON ) {     
    if( 'modules' in payload ) {
      addModules(payload['modules']);
    }
    
    if( 'plugin_types' in payload ) {
      addPluginTypes(payload['plugin_types']);
    }
    
    if( 'plugin_added' in payload ) {
      addPlugins(payload['plugin_added']);
    }
    
    if( 'plugin_connected' in payload ) {
      connectPlugins(payload['plugin_connected']);
    }
    
    if( 'plugin_removed' in payload ) {
      removePlugins(payload['plugin_removed']);
    }
    
    if( 'send_output' in payload ) {
      sendOutput(payload['send_output']);
    }
    
    if( 'state_dict' in payload ) {
      setStateDict(payload['state_dict']);
    }
    
    if( 'stats' in payload ) {
      setStats(payload['stats']);
    }
    
    if( 'presets' in payload ) {
      setPresets(payload['presets']);
    }
    
    if( 'alert' in payload ) {
      addAlert(payload['alert']);
    }
  }
  else if( type == MESSAGE_AUDIO ) {
    onAudioOutput(payload);
  }
}

function removeAll(name) {
  sendWebsocket({'reset': {'plugins': true, 'globals': true}});
}

function clearCache(name) {
  sendWebsocket({'clear_cache': {}})
}

function savePreset(name) {
  sendWebsocket({'save': name});
}

function loadPreset(name) {
  sendWebsocket({'load': name});
}

function insertPreset(name, context=true) {
  if( context ) {
    sendWebsocket({
      'insert': {
        'path': name, 
        'layout_node': nodeLayoutInsertPos()
      }
    });
  }
  else {
    sendWebsocket({'insert': name});
  }
}

function setPresets(presets) {
  console.log('setting presets list', presets);
  
  let load_menu = $('#navbarLoadAgentMenu');
  let insert_menu = $('#navbarInsertAgentMenu');
  let context_menu = $('#plugin_context_menu_presets');
  
  load_menu.empty();
  insert_menu.empty();
  context_menu.empty();
  
  presets.forEach((preset) => {
    load_menu.append(`<li><a class="dropdown-item" href="#" onclick="loadPreset('${preset}')">${preset}</a></li>`);
    insert_menu.append(`<li><a class="dropdown-item" href="#" onclick="insertPreset('${preset}')">${preset}</a></li>`);
    context_menu.append(`<li><a class="dropdown-item" href="#" onclick="insertPreset('${preset}', true)">${preset}</a></li>`);
  });
}

// https://stackoverflow.com/a/77965966
/*import './webrtc.js';
import * as websocket from './websocket.js';
import './debounce.js';
import './audio.js';
import './grid.js';
import './menu.js';

Object.assign(globalThis, websocket);
console.log('globalThis', globalThis);
*/

// https://ardislu.dev/import-javascript-module-into-global
/*await Promise.all([
  import('./webrtc.js'),
  import('./websocket.js'),
  import('./debounce.js'),
  import('./audio.js'),
  import('./grid.js'),
  import('./menu.js')
]).then(modules => modules.forEach(m => Object.assign(globalThis, m)));

console.log('globalThis', globalThis);*/
