
var grid;
var drawflow;
var pluginTypes;
var nodeIdToName = {};
var stateListeners = {};
var ignore_connection_events=false;


function addGrid() {
  grid = GridStack.init({'column': 12, 'cellHeight': 50, 'float': true});

  grid.on('added', function(event, items) {
    items.forEach((item) => {
      console.log(`grid widget ${item.id} was added`, item);
      sendWebsocket({
        'config_plugin': {
          'name': item.id,
          'args': {'web_grid': {'x': item.x, 'y': item.y, 'w': item.w, 'h': item.h}} 
      }});
    });
  });
  
  grid.on('change', function(event, items) {
    items.forEach((item) => {
      console.log(`grid widget ${item.id} changed position/size`, item);
      sendWebsocket({
        'config_plugin': {
          'name': item.id,
          'args': {'web_grid': {'x': item.x, 'y': item.y, 'w': item.w, 'h': item.h}} 
      }});
    });
  });
  
  grid.on('removed', function(event, items) {
    items.forEach((item) => {
      console.log(`grid widget ${item.id} has been removed`, item);
      sendWebsocket({
        'config_plugin': {
          'name': item.id,
          'args': {'web_grid': {}} 
      }});
    });
  });
  
  
  /*grid.on('resize', function(event, el) {
    fitGridWidgetContents(el);
  });*/

  /*addGraphEditor();

  drawflow = new Drawflow(document.getElementById("drawflow"));
  drawflow.start();
  
  drawflow.on('connectionCreated', onNodeConnectionCreated);
  drawflow.on('connectionRemoved', onNodeConnectionRemoved);*/
}

function addGridWidget(id, title, html, titlebar_html, grid_options) {
  const plugin = id.includes('_grid') ? id.replace('_grid', '') : id;
  
  if( grid_options == undefined ) 
      grid_options = {w: 3, h: 3};

  if( titlebar_html != undefined )
      titlebar_html = `<span class="float-top">${titlebar_html}</span>`;
   
  let title_html = '';
  
  if( title != undefined || titlebar_html != undefined ) {
    title_html += `<span class="mb-2">`;
    
    if( title != undefined )
      title_html += `<h5 class="d-inline">${title}</h5>`;

    title_html += `<button id="${id}_close" type="button" class="btn-close float-end float-top ms-2" aria-label="Close"></button>`;
    
    if( document.getElementById(`${title}_config_dialog`) != null )
      title_html += `<i id="${id}_show_config" class="fa fa-cog float-end gear-button" aria-hidden="true"></i>`;
    
    if( titlebar_html != undefined )
      title_html += titlebar_html;

    title_html += `</span>`;
  }    
  //  position: absolute; right: 15px;
  /*let card_html = `
  <div class="card ms-1 mt-1">
    <div class="card-body d-flex flex-column h-100">
      <div class="card-title" data-bs-toggle="collapse" href="${id}" role="button">
        <h5 class="d-inline">${title}</h5>
        <span class="float-end float-top fa fa-chevron-circle-down fa-lg mt-1 ms-1 me-1" data-bs-toggle="collapse" href="#${id}" id="${id}_collapse_btn" title="Minimize"></span>
        ${titlebar_html}
      </div>
      <div class="collapse show d-flex flex-column h-100" id="${id}">
          ${html}
      </div>
    </div>
  </div>
  `;*/

  let card_html = `
  <div class="card" id="${id}">
    <div class="card-body d-inline-flex flex-column h-100">
      ${title_html}
      ${html}
    </div>
  </div>
  `;
  
  if( ! ('id' in grid_options) )
    grid_options['id'] = plugin;
    
  const widget = grid.addWidget(card_html, grid_options);
  
  console.log(`created grid widget id=${id} title=${title}`, widget);
  
  let btn_close = document.getElementById(`${id}_close`);
  
  if( btn_close != undefined ) {
      btn_close.addEventListener('click', function(e) {
      grid.removeWidget(widget);
    });
  }
  
  let btn_config = document.getElementById(`${id}_show_config`);
  
  if( btn_config != undefined ) {
      btn_config.addEventListener('click', function(e) {
        sendWebsocket({'get_state_dict': title});
        const config_modal = new bootstrap.Modal(`#${title}_config_dialog`);
        config_modal.show();
    });
  }
  
  addStateListener(plugin, function(state_dict) {
    if( 'web_grid' in state_dict ) {
      console.log(`updating ${plugin} grid widget`, state_dict['web_grid']);
      grid.update(widget, state_dict['web_grid']);
    }
  });
  
  sendWebsocket({'get_state_dict': plugin});
  return widget;
} 

function addTextInputWidget(name, id, grid_options) {
  const input_id = `${id}_input`;
  const submit_id = `${id}_submit`;
  
  const html = `
    <div class="input-group">
      <textarea id="${input_id}" class="form-control" rows="2" placeholder="Enter to send (Shift+Enter for newline)"></textarea>
      <span id="${submit_id}" class="input-group-text bg-light-gray bi bi-arrow-return-left" style="color: #eeeeee;"></span>
    </div>
  `;
  
  let widget = addGridWidget(id, null, html, null, Object.assign({w: 4, h: 2}, grid_options));

  let onsubmit = function() {
    const input = document.getElementById(input_id);
    console.log('submitting text input', input.value);
    msg = {}
    msg[name] = {'input': input.value};
    sendWebsocket(msg);
    input.value = "";
  }
  
  let onkeydown = function(event) {
    // https://stackoverflow.com/a/49389811
    if( event.which === 13 && !event.shiftKey ) {
      if( !event.repeat )
        onsubmit();
      event.preventDefault(); // prevents the addition of a new line in the text field
    }
  }

  document.getElementById(input_id).addEventListener('keydown', onkeydown);
  document.getElementById(submit_id).addEventListener('click', onsubmit);

  return widget;
}

function addTextStreamWidget(name, id, grid_options) {
  const history_id = `${id}_history`;

  const html = `
    <div id="${history_id}" class="bg-medium-gray p-2 mb-2" style="font-family: monospace, monospace; font-size: 100%; overflow-y: scroll; flex-grow: 1;"</div>
  `;
  
  let widget = addGridWidget(id, name, html, null, Object.assign({w: 4, h: 6}, grid_options));

  addStateListener(name, function(state_dict) {
    if( ! ('text' in state_dict) )
      return;
    
    let el_type = 'p';  
    
    if( 'delta' in state_dict )
      el_type = 'span';

    document.getElementById(history_id).innerHTML += `
      <${el_type} style="color: ${state_dict['color']}">${state_dict['text']}</${el_type}>
    `;
  });
  
  return widget;
}

function addChatWidget(name, id, grid_options) {
  const history_id = `${id}_history`;
  const input_id = `${id}_input`;
  const submit_id = `${id}_submit`;
  
  const html = `
    <div id="${history_id}" class="bg-medium-gray p-2 mb-2" style="font-size: 100%; overflow-y: scroll; flex-grow: 1;" ondrop="onFileDrop(event)" ondragover="onFileDrag(event)"></div>
    <div class="input-group">
      <textarea id="${input_id}" class="form-control" rows="2" placeholder="Enter to send (Shift+Enter for newline)"></textarea>
      <span id="${submit_id}" class="input-group-text bg-light-gray bi bi-arrow-return-left" style="color: #eeeeee;"></span>
    </div>
  `;
  
  let widget = addGridWidget(id, name, html, null, Object.assign({w: 4, h: 14}, grid_options));

  let onsubmit = function() {
    const input = document.getElementById(input_id);
    console.log('submitting chat message:', input.value);
    sendWebsocket({'UserPrompt': {'input': input.value}});
    input.value = "";
  }
  
  let onkeydown = function(event) {
    // https://stackoverflow.com/a/49389811
    if( event.which === 13 && !event.shiftKey ) {
      if( !event.repeat )
        onsubmit();
      event.preventDefault(); // prevents the addition of a new line in the text field
    }
  }

  document.getElementById(input_id).addEventListener('keydown', onkeydown);
  document.getElementById(submit_id).addEventListener('click', onsubmit);

  addStateListener(name, function(state_dict) {
    console.log(`${name}.stateListener()`, state_dict);
    if( 'history' in state_dict ) {
      updateChatHistory(history_id, state_dict['history']);
    }
  });
  
  sendWebsocket({'get_state_dict': name});
  return widget;
}

function updateChatHistory(id, history) {    
    let chj = $(`#${id}`);
    let chc = document.getElementById(id);
    let isScrolledToBottom = chc.scrollHeight - chc.clientHeight <= chc.scrollTop + 1;

    chj.empty(); // clear because server may remove partial/rejected ASR prompts
    
    for( let n=0; n < history.length; n++ ) {
      const role = history[n]['role'];
      
      /*if( role == 'system' )
        continue;*/
        
      let contents = '';
      var hasImage = 'image' in history[n];
      
      if( hasImage ) {
        contents += `<img src=${history[n]['image']} width="100%">`;
        imageAtBottom = true;
      }
      
      if( 'text' in history[n] )
        contents += history[n]['text'];

      if( contents.length > 0 )
        chj.append(`<div id="msg_${n}" class="chat-message-${role} mb-3">${contents}</div><br/>`);
    }

    function scrollBottom(container) {  // https://stackoverflow.com/a/21067431
      container.scrollTop = container.scrollHeight - container.clientHeight;
      console.log(`scrolling to bottom ${container.scrollTop} ${container.scrollHeight} ${container.clientHeight}`);
    }

    if( isScrolledToBottom ) { // autoscroll unless the user has scrolled up
      if( hasImage )
        setTimeout(scrollBottom, 50, chc);  // wait for images to load to get right height
      else
        scrollBottom(chc);
    }
}


function addVideoOutputWidget(name, id, grid_options) {
  const video_id = `${id}_video_player`;
  const html = `
    <div>
      <video id="${video_id}" autoplay controls playsinline muted>Your browser does not support video</video>
    </div>
  `;
  
  let widget = addGridWidget(id, name, html, null, Object.assign({w: 8, h: 5}, grid_options));
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


function addStateListener(name, listener) {
  if( name == undefined )
    name = 'all';
    
  if( ! (name in stateListeners) )
    stateListeners[name] = [listener];
  else
    stateListeners[name].push(listener);
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
    
    if( plugin_name in stateListeners )
      stateListeners[plugin_name].forEach((listener) => {listener(state_dict);});

    if( 'all' in stateListeners )
      stateListeners['all'].forEach((listener) => {listener(state_dict);});
  }
}

function setStats(stats_dicts) {
  for( plugin_name in stats_dicts ) {
    const stats = stats_dicts[plugin_name];
    
    if( ! ('summary' in stats) )
      continue;
    
    let summary = stats['summary'];
    
    if( Array.isArray(summary) ) {
      const num_outputs = $(`.${plugin_name} .outputs .output`).length;
      summary = summary.join($(`.${plugin_name}`).length == 0 || num_outputs > 0 ? '<br/>' : ' ');
    }
    
    let node_stats = document.getElementById(`${plugin_name}_node_stats`);
    
    if( node_stats != undefined ) 
      node_stats.innerHTML = summary;
  }
}


function addPlugin(plugin) {
  console.log('addPlugin() =>');
  console.log(plugin)
  
  const plugin_name = plugin['name'];
  const web_grid = plugin['web_grid'];
  const web_node = plugin['web_node'];
  const nodes = document.querySelectorAll('.drawflow-node');

  if( ('x' in web_node) && ('y' in web_node) ) {
    var x = web_node['x'];
    var y = web_node['y'];
  }
  else
  {
    var x = 10;
    var y = 10;
      
    if( nodes.length > 0 ) {
      const node = nodes[nodes.length-1];
      const abs_rect = node.getBoundingClientRect();
      x = node.offsetLeft + abs_rect.width + 60;
      y = node.offsetTop
    }
  }
  
  let stats_class = '';
  
  if( plugin['outputs'].length > 1 )
    stats_class = 'mt-2';
    
  const html = `
    <div style="position: absolute; top: 5px;">
      ${plugin_name}
      <p id="${plugin_name}_node_stats" class="${stats_class}" style="font-family: monospace, monospace; font-size: 80%"></p>
    </div>
  `;
  
  const node_id = drawflow.addNode(plugin_name, plugin['inputs'].length, plugin['outputs'].length, x, y, plugin_name, {}, html);
  nodeIdToName[node_id] = plugin_name;
  console.log(`added node id=${node_id} for {plugin_name}`);
  
  let style = '';
  let max_output_chars = 0;
  
  for( i in plugin['outputs'] ) {
    const output = plugin['outputs'][i].toString();
    max_output_chars = Math.max(max_output_chars, output.length);
    style += `.${plugin_name} .outputs .output:nth-child(${Number(i)+1}):before {`;
    style += `display: block; content: "${output}"; position: relative; min-width: 160px; font-size: 80%; bottom: 2px; right: ${(output.length-1) * 6 + 20}px;} `;
  }
    
  style += `.outputs { margin-top: 20px; font-family: monospace, monospace; } `;
  
  var style_el = document.createElement('style');
  style_el.id = `${plugin_name}_node_io_styles`;
  style_el.innerHTML = style;
  document.head.appendChild(style_el);

  const has_config_dialog = (Object.keys(plugin['parameters']).length > 0);
  
  $(`.${plugin_name}`).on('dblclick', function () {
    console.log(`double-click ${plugin_name}`);
    if( addPluginGridWidget(plugin_name, plugin['type']) == null ) {
      if( has_config_dialog ) {
        sendWebsocket({'get_state_dict': plugin_name});
        const config_modal = new bootstrap.Modal(`#${plugin['name']}_config_dialog`);
        config_modal.show();
      }
    }
  });
  
  if( has_config_dialog )
    addPluginDialog(plugin_name, 'config', null, plugin['parameters']);
    
  if( Object.keys(web_grid).length > 0 ) {
    addPluginGridWidget(plugin_name, plugin['type'], web_grid);
  }
}

function addPluginGridWidget(name, type, grid_options) {
  const id = `${name}_grid`;
  
  if( type == undefined )
    type = name;
    
  if( document.getElementById(id) != null )
    return null;
    
  switch(type) {
    case 'UserPrompt':
      return addTextInputWidget(name, id, grid_options);
    case 'TextStream':
      return addTextStreamWidget(name, id, grid_options);
    case 'ChatSession':
      return addChatWidget(name, id, grid_options);
    case 'VideoOutput':
      return addVideoOutputWidget(name, id, grid_options);
    case 'GraphEditor':
      return addGraphEditor(name, id, grid_options);
    default:
      return null;
  }
}

function addPlugins(plugins) {
  if( !Array.isArray(plugins) )
    plugins = [plugins];
    
  for( i in plugins ) {
    let plugin = plugins[i];
    if( ('global' in plugin) ) {
      addPluginGridWidget(plugin['name'], null, ('web_grid' in plugin) ? plugin['web_grid'] : null);
    }
    else
      addPlugin(plugin);
  }
  
  ignore_connection_events = true;
  
  for( i in plugins ) {
    for( l in plugins[i]['connections'] ) {
      const conn = plugins[i]['connections'][l];
      console.log('adding connection', conn);

      drawflow.addConnection(
        drawflow.getNodesFromName(plugins[i]['name'])[0], 
        drawflow.getNodesFromName(conn['to'])[0],  
        `output_${conn['output']+1}`, 
        `input_${conn['input']+1}`
      );
    }
  }
  
  ignore_connection_events = false;
}

function addPluginTypes(types) {
  pluginTypes = types;
  
  /*let id = document.getElementById("add_plugin_menu_list");
            
  if( id != null )
    id.innerHTML = pluginMenuList();*/

  for( pluginName in pluginTypes ) {
    const plugin = pluginTypes[pluginName];

    if( 'init' in plugin && Object.keys(plugin['init']['parameters']).length > 0 ) {
      addPluginDialog(pluginName, 'init', plugin['init']['description'], plugin['init']['parameters']);
    }
  }
}

function pluginMenuList() {
  if( pluginTypes == undefined )
    return '';
 
  let html = '';
  
  for( plugin in pluginTypes )
    html += `<li><a class="dropdown-item" id="menu_create_plugin_${plugin}" href="#">${plugin}</a></li>`; // data-bs-toggle="modal" data-bs-target="#${plugin}_init_dialog"

  return html; 
}  

function addGraphEditor(name, id, grid_options) {

  let titlebar_html = `
    <a class="dropdown-toggle ms-2" href="#" data-bs-toggle="dropdown" aria-expanded="false" style="color: #aaaaaa;">
      Add
    </a>
    <span class="dropdown float-end float-top">
      <ul class="dropdown-menu" id="add_plugin_menu_list">
        ${pluginMenuList()}
      </ul>
    </span>
  `;
  
  let html = `
    <div id="drawflow" class="mt-1 flex-grow-1 w-100"></div>
  `;

  let widget = addGridWidget(
     id, "Graph Editor", 
     html, titlebar_html, 
     Object.assign({w: 8, h: 11}, grid_options)
  );
  
  let editor = document.getElementById("drawflow");
  
  editor.addEventListener('mouseenter', (event) => {
    console.log('onmouseleave(drawflow) => disabling grid move/resize');
    grid.disable();
  });
  
  editor.addEventListener('mouseleave', (event) => {
    console.log('onmouseleave(drawflow) => enabling grid move/resize');
    grid.enable();
  });
  
  drawflow = new Drawflow(document.getElementById("drawflow"));
  drawflow.start();
  
  drawflow.on('nodeMoved', onNodeMoved);
  drawflow.on('nodeRemoved', onNodeRemoved);
  
  drawflow.on('connectionCreated', onNodeConnectionCreated);
  drawflow.on('connectionRemoved', onNodeConnectionRemoved);
  
  for( pluginName in pluginTypes ) {
    const plugin = pluginTypes[pluginName];
    let pluginMenu = document.getElementById(`menu_create_plugin_${pluginName}`);
    
    if( 'init' in plugin && Object.keys(plugin['init']['parameters']).length > 0 ) {
      pluginMenu.addEventListener('click', (event) => {
        console.log(`opening init dialog for ${plugin['name']}`);
        const modal = new bootstrap.Modal(`#${plugin['name']}_init_dialog`);
        modal.show();
      });
    }
    else {
      pluginMenu.addEventListener('click', (event) => {
        sendWebsocket({
          'init_plugin': {
            'name': plugin['name'],
            'args': {}
          }
        });
      });
    }
  }
  
  return widget;
}

function onNodeMoved(id) {
  const node = drawflow.getNodeFromId(id);
  console.log('node moved', id, node);
  sendWebsocket({
    'config_plugin': {
      'name': node['name'],
      'args': {'web_node': {'x': node['pos_x'], 'y': node['pos_y']}} 
  }});
}

function onNodeRemoved(id) {
  const pluginName = nodeIdToName[id];
  delete nodeIdToName[id];
  console.log(`node removed id=${id} name=${pluginName}`);
  sendWebsocket({
    'remove_plugin': pluginName,
  });
}

function onNodeConnectionCreated(event) {
  console.log(`onNodeConnectionCreated(ignore=${ignore_connection_events})`, event);
  
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

function onNodeConnectionRemoved(event) {
  console.log(`onNodeConnectionRemoved(ignore=${ignore_connection_events})`, event);
  
  if( ignore_connection_events )
    return;
    
  sendWebsocket({
    'remove_connection': {
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

function addDialog(id, title, html, xl, onsubmit, oncancel) {
  $('body').append(`
  <div class="modal fade" id="${id}" tabindex="-1" aria-labelledby="${id}_label" aria-hidden="true">
      <div class="modal-dialog ${xl ? 'modal-xl' : ''}">
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

function addPluginDialog(plugin_name, stage, description, parameters, max_per_column) {
  if( max_per_column == undefined )
    max_per_column = 6;
    
  const num_params = Object.keys(parameters).length;
  const num_columns = Math.ceil(num_params / max_per_column);
  
  const dialog_id = `${plugin_name}_${stage}_dialog`;
  const dialog_xl = (num_params > max_per_column);
  
  let html = '';
  
  if( description != null )
    html += `<p>${description}</p>`;
    
  html += `<div class="container">`;
  html += `<div class="row">`;
  html += `<div class="col-sm">`;

  var select2_args = {};
  let param_count = 0;
  
  for( param_name in parameters ) {
    const param = parameters[param_name];
    const id = `${plugin_name}_${stage}_${param_name}`;
    
    if( param['hidden'] )
      continue;
    
    if( param_count > 0 && param_count % Math.ceil(num_params / num_columns) == 0 )
      html += `</div><div class="col-sm">`;
      
    let value = '';
    let value_html = '';
    
    if( 'default' in param && param['default'] != undefined ) {
      value = param['default'];
      value_html = `value="${value}"`;
    }
     
    let help = '';
    
    if( 'help' in param )
      help = `<div id="${id}_help" class="form-text">${param['help']}</div>`;
    
    switch(param['type']) {
      case 'number':
      case 'integer':
        var type='number'; break;
      default:
        var type='text';
    }
    
    if( 'options' in param /*|| has_suggestions*/ ) {
      //if( has_options ) {
      let options = param['options'];
      //}
      /*else if( has_suggestions ) {
        var options = param['suggestions'];
        select2_args[id] = {tags: true, placeholder: 'enter'};
      }*/
      
      var input_html = `<select id="${id}" class="form-control">\n`
      
      for( i in options ) {
        if( i == 0 )
          var selected = ` selected="selected"`;
        else
          var selected = '';
        
        input_html += `  <option${selected}>${options[i]}</option>\n`
      }
      
      input_html += `</select>\n`;
    }
    else if( 'suggestions' in param ) {
      const list_id = `${id}_list`;
      var input_html = `<input id="${id}" type="${type}" class="form-control" list="${list_id}"/>`;
      
      input_html += `<datalist id="${list_id}">`;
      
      for( i in param['suggestions'] ) {
        input_html += `<option>${param['suggestions'][i]}</option>`;
      }
      
      input_html += `</datalist>`; 
    }
    else if( 'multiline' in param ) {
      var input_html = `<textarea id="${id}" class="form-control" aria-describedby="${id}_help" rows=${param['multiline']}>${value}</textarea>`;
    }
    else if( 'color' in param ) {
      var input_html = `<input id="${id}" type="color" class="form-control" aria-describedby="${id}_help" ${value_html}/>`;
    }
    else {
      var input_html = `<input id="${id}" type="${type}" class="form-control" aria-describedby="${id}_help" ${value_html}>`;
    }

    html += `
      <div class="mb-3">
        <label for="${id}" class="form-label">${param['display_name']}</label>
        ${input_html}
        ${help}
      </div>
    `;
    
    param_count += 1;
  }
  
  html += `</div></div></div>`;
  
  let onsubmit = function() {
    console.log(`onsubmit(${plugin_name})`);
    let args = {};
    
    for( param_name in parameters ) {
      let value = document.getElementById(`${plugin_name}_${stage}_${param_name}`).value;
      if( value != undefined && value.length > 0 ) { // input.value are always strings
        const type = parameters[param_name]['type'];
        if( type == 'integer' || type == 'number' )
          value = Number(value);
        args[param_name] = value;
      }
    }
    
    console.log(`${stage}Plugin(${plugin_name}) =>`);
    console.log(args);

    let msg = {};
    
    msg[`${stage}_plugin`] = {
        'name': plugin_name,
        'args': args
    }
    
    sendWebsocket(msg);
  }
  

  addDialog(dialog_id, plugin_name, html, dialog_xl, onsubmit);
  
  //for( select2_id in select2_args )
  //  $(`#${select2_id}`).select2(select2_args[select2_id]);

  return dialog_id;
}

function fitGridWidgetContents(el) {
  //let width = parseInt(el.getAttribute('gs-w')) || 0;
  //let height = parseInt(el.getAttribute('gs-h')) || 0;
  //console.log('grid widget resize', width, height, el.offsetWidth, el.offsetHeight);
  //let card_title = el.querySelector('.card-title');
  //console.log('grid widget resize', card_title, card_title.offsetHeight);
  //el.querySelector('.collapse').style.height = `${el.offsetHeight - card_title.offsetHeight - 50}px`;
  console.log('grid widget resize', el.offsetWidth, el.offsetHeight);
  
  let grid_rect = el.getBoundingClientRect();

  let card = el.querySelector('.card-body');
  let card_rect = card.getBoundingClientRect();
  
  let card_inner = el.querySelector('.collapse');
  let card_inner_rect = card_inner.getBoundingClientRect();
  
  console.log(`grid widget rect height=${grid_rect.height}  bottom=${grid_rect.bottom}`);
  console.log(`card widget rect height=${card_rect.height}  bottom=${card_rect.bottom}`);
  console.log(`card inner rect height=${card_inner_rect.height}  bottom=${card_inner_rect.bottom}`);
  
  //if( card_inner_rect.bottom > card_rect.bottom ) {
    //card_inner.style.height = `${card_inner_rect.height - (card_inner_rect.bottom - card_rect.bottom)}px`;
    //console.log('set height', card_inner.style.height);
  //}
  /*if( card_rect.height > grid_rect.height ) {
    //card.style.height = `${grid_rect.height}px`;
    card_inner.style.height = `${card_inner.offsetHeight - (card_rect.height - grid_rect.height)}px`;
  }*/
  //card.style.height = `${el.offsetHeight}px`;
  //el.querySelector('.collapse').style.height = `${el.offsetHeight-150}px`;
}


