
var grid;
var drawflow;
var plugin_types;

function addGridWidget(id, title, html, titlebar_html, grid_options) {
  if( grid_options == undefined ) 
      grid_options = {w: 3, h: 3};
    
  if( title == undefined ) 
      title = name;

  if( titlebar_html == undefined )
      titlebar_html = '';
  else
      titlebar_html = `<span class="float-end float-top">${titlebar_html}</span>`;
      
  let card_html = `
  <div class="card ms-1 mt-1">
    <div class="card-body">
      <div class="card-title" data-bs-toggle="collapse" href="${id}" role="button">
        <h5 class="d-inline">${title}</h5>
        <span class="float-end float-top fa fa-chevron-circle-down fa-lg mt-1 ms-1 me-1" data-bs-toggle="collapse" href="#${id}" id="${id}_collapse_btn" title="Minimize"></span>
        ${titlebar_html}
      </div>
      <div class="collapse show" id="${id}">
          ${html}
      </div>
    </div>
  </div>
  `
  
  return grid.addWidget(card_html, grid_options);
} 

function addGrid() {
  grid = GridStack.init({'column': 12, 'cellHeight': 50});

  addGraphEditor();
  addGridWidget("test_widget", "Test", `ABC123`);

  //grid.addWidget({w: 2, content: 'item 1'});
  //grid.addWidget({w: 3, content: 'hello'});

  drawflow = new Drawflow(document.getElementById("drawflow"));
  drawflow.start();
  
  drawflow.on('connectionCreated', onConnectionCreated);
}

var ignore_connection_events=false;

function onConnectionCreated(event) {
  console.log(`onConnectionCreated(ignore=${ignore_connection_events})`, event);
  
  if( ignore_connection_events )
    return;
    
  sendWebsocket({
    'add_connection': {
      'input': {
        'name': drawflow.getNodeFromId(event['input_id'])['name'],
        'channel': Number(event['input_class'].split('_').at(-1)) - 1
      },
      'output': {
        'name': drawflow.getNodeFromId(event['output_id'])['name'],
        'channel': Number(event['output_class'].split('_').at(-1)) - 1
      }
    }
  });
}

function pluginMenuList() {
  if( plugin_types == undefined )
    return '';
 
  let html = '';
  
  for( plugin in plugin_types )
    html += `<li><a class="dropdown-item" href="#" data-bs-toggle="modal" data-bs-target="#${plugin}_init_dialog">${plugin}</a></li>`;

  return html; 
}  

function addPluginTypes(types) {
  plugin_types = types;
  
  let id = document.getElementById("add_plugin_menu_list");
            
  if( id != null )
    id.innerHTML = pluginMenuList();
    
  for( plugin_name in plugin_types ) {
    const plugin = plugin_types[plugin_name];
    
    if( 'init' in plugin )
      addPluginDialog(plugin, 'init');
      
    if( 'config' in plugin )
      addPluginDialog(plugin, 'config');
  }
}

function setStateDict(state_dicts) {
  for( plugin_name in state_dicts ) {
    const state_dict = state_dicts[plugin_name];
    for( state_name in state_dict ) {
      const id = document.getElementById(`${plugin_name}_config_${state_name}`);
      if( id != null ) {
        id.value = state_dict[state_name];
      }
    }
  }
}

function addPlugin(plugin) {
  console.log('addPlugin() =>');
  console.log(plugin)
  
  const plugin_name = plugin['name'];
  const nodes = document.querySelectorAll('.drawflow-node');
  
  let x = 10;
  let y = 10;
    
  if( nodes.length > 0 ) {
    const node = nodes[nodes.length-1];
    const abs_rect = node.getBoundingClientRect();
    x = node.offsetLeft + abs_rect.width + 45;
    y = node.offsetTop
  }

  drawflow.addNode(plugin_name, plugin['inputs'].length, plugin['outputs'].length, x, y, plugin_name, {}, `<div>${plugin_name}</div>`);

  $(`.${plugin_name}`).on('dblclick', function () {
    console.log(`double-click ${plugin_name}`);
    if( addPluginGridWidget(plugin_name, plugin['type']) == null ) {
      sendWebsocket({'get_state_dict': plugin_name});
      const config_modal = new bootstrap.Modal(`#${plugin['name']}_config_dialog`);
      config_modal.show();
    }
  });
}

function addPluginGridWidget(name, type) {
  const id = `${name}_grid`;
  
  if( document.getElementById(id) != null )
    return null;
    
  switch(type) {
    case 'VideoOutput':
      return addVideoOutputWidget(name, id);
    default:
      return null;
  }
}

function addVideoOutputWidget(name, id) {
  const video_id = `${id}_video_player`;
  const html = `
    <div>
      <video id="${video_id}" autoplay controls playsinline muted>Your browser does not support video</video>
    </div>
  `;
  
  let widget = addGridWidget(id, name, html, null, {x: 0, y: 0, w: 8, h: 5});
  let video = document.getElementById(video_id);
  
  video.addEventListener('playing', function(e) {
    const abs_rect = video.getBoundingClientRect();
    const options = {'w': Math.ceil(abs_rect.width/grid.cellWidth()), 'h': Math.ceil(abs_rect.height/grid.getCellHeight())+1};
    console.log(`${video_id} playing`, e, abs_rect, options, grid.cellWidth(), grid.getCellHeight());
    grid.update(widget, options);
  });

  playStream(getWebsocketURL('output'), video); // TODO handle actual stream name
  
  return widget;
}

function addPlugins(plugins) {
  if( !Array.isArray(plugins) )
    plugins = [plugins];
    
  for( i in plugins )
    addPlugin(plugins[i]);

  ignore_connection_events = true;
  
  for( i in plugins ) {
    for( l in plugins[i]['links'] ) {
      const link = plugins[i]['links'][l];
      console.log('adding connection', link);

      drawflow.addConnection(
        drawflow.getNodesFromName(plugins[i]['name'])[0], 
        drawflow.getNodesFromName(link['to'])[0],  
        `output_${link['output']+1}`, 
        `input_${link['input']+1}`
      );
    }
  }
  
  ignore_connection_events = false;
}


function addGraphEditor() {
  let html = `
    <div id="drawflow" class="mt-3"></div>
  `;
  
  let titlebar_html = `
    <div class="dropdown">
      <button class="btn btn-secondary dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false">
        Add
      </button>
      <ul class="dropdown-menu" id="add_plugin_menu_list">
        ${pluginMenuList()}
      </ul>
    </div>
  `;
  
  return addGridWidget("graph_editor", "Graph Editor", html, titlebar_html, {x: 0, y: 0, w: 8, h: 11, noMove: true});
}

function addDialog(id, title, html, onsubmit, oncancel) {
  $('body').append(`
  <div class="modal fade" id="${id}" tabindex="-1" aria-labelledby="${id}_label" aria-hidden="true">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="${id}_label">${title}</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close" style="color: #eeeeee;"></button>
          </div>
          <div class="modal-body">
            ${html}
          </div>
          <div class="modal-footer">
            <button id="${id}_cancel" type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button> 
            <button id="${id}_submit" type="button" class="btn btn-secondary" data-bs-dismiss="modal">Okay</button>
          </div>
        </div>
      </div>
    </div>
  `);
  
  if( onsubmit != undefined )
    $(`#${id}_submit`).on('click', onsubmit);
    
  if( oncancel != undefined )
    $(`#${id}_cancel`).on('click', oncancel);
}

function addPluginDialog(plugin, stage) {

  let plugin_name = plugin['name'];
  
  let html = '';
  
  if( stage != 'config' )
    html += `<p>${plugin[stage]['description']}</p>`;
    
  html += '<form>';

  for( param_name in plugin[stage]['parameters'] ) {
    const param = plugin[stage]['parameters'][param_name];
    const id = `${plugin['name']}_${stage}_${param_name}`;
    
    let value = '';
    
    if( 'default' in param && param['default'] != undefined )
      value = `value="${param['default']}"`;
      
    let help = '';
    
    if( 'description' in param )
      help = `<div id="${id}_help" class="form-text">${param['description']}</div>`;
    
    switch(param['type']) {
      case 'number':
      case 'integer':
        var type='number'; break;
      default:
        var type='text';
    }
    
    html += `
      <div class="mb-3">
        <label for="${id}" class="form-label">${param['display_name']}</label>
        <input id="${id}" type="${type}" class="form-control" aria-describedby="${id}_help" ${value}>
        ${help}
      </div>
    `;
  }
  
  html += `</form>`;
  
  let onsubmit = function() {
    console.log(`onsubmit(${plugin['name']})`);
    let args = {};
    
    for( param_name in plugin[stage]['parameters'] ) {
      let value = document.getElementById(`${plugin['name']}_${stage}_${param_name}`).value;
      if( value != undefined && value.length > 0 ) { // input.value are always strings
        const type = plugin[stage]['parameters'][param_name]['type'];
        if( type == 'integer' || type == 'number' )
          value = Number(value);
        args[param_name] = value;
      }
    }
    
    console.log(`${stage}Plugin(${plugin['name']}) =>`);
    console.log(args);

    let msg = {};
    
    msg[`${stage}_plugin`] = {
        'type': plugin['name'],
        'args': args
    }
    
    sendWebsocket(msg);
  }
  
  addDialog(`${plugin['name']}_${stage}_dialog`, plugin['name'], html, onsubmit);
}

